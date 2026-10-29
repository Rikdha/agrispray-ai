#!/usr/bin/env python3
"""
detector_node.py
Runs the YOLOv8 PlantDoc model (best.onnx) on camera frames, draws boxes,
assigns a pesticide + dosage to each detection and publishes the result.

Subscribes : /camera/image/compressed        (sensor_msgs/CompressedImage)
Publishes  : /detector/image_annotated/compressed (sensor_msgs/CompressedImage)
             /detections                     (std_msgs/String, JSON)

!! The pesticide names / amounts below are EXAMPLE PLACEHOLDERS. Check the
!! product label, local regulations and an agronomist before real use, and
!! calibrate PUMP_FLOW_ML_PER_S for your own pump.
"""
import ast
import json
import math
import os
import re
import threading
import time
import traceback
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# Fallback class names (PlantDoc, alphabetical). The node first tries to read
# the names stored inside best.onnx - VERIFY against your data.yaml if the
# model has no embedded names.
DEFAULT_NAMES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
    'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf',
    'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf',
    'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf',
    'Soyabean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'Tomato two spotted spider mites leaf', 'grape leaf', 'grape leaf black rot',
]

# (keyword in the NORMALISED class name, group, pesticide, base dose in ml per plant)
# First match wins; keywords match whole words, case/underscores ignored.
# No match (e.g. 'healthy') => no spray.   EXAMPLE VALUES ONLY - check the product label / agronomist.
_FUNG = 'Mancozeb / chlorothalonil fungicide'
_VIRAL = 'NONE - viral disease, remove plant / control vectors'
TREATMENTS = [
    ('fungal',          'fungal',    'Broad-spectrum fungicide (e.g. mancozeb / chlorothalonil)', 10.0),
    ('bacterial',       'bacterial', 'Copper-based bactericide',                                  10.0),
    ('mosaic',          'viral',     _VIRAL,                                                       0.0),
    ('virus',           'viral',     _VIRAL,                                                       0.0),
    ('spider mites',    'pest',      'Miticide (e.g. neem oil / abamectin)',                       8.0),
    ('pepper leaf spot', 'bacterial', 'Copper-based bactericide',                                 10.0),
    ('powdery mildew',  'fungal',    'Sulfur / potassium bicarbonate fungicide',                   8.0),
    ('rust',            'fungal',    'Triazole fungicide (e.g. propiconazole)',                   10.0),
    ('late blight',     'fungal',    'Metalaxyl + mancozeb fungicide',                            12.0),
    ('blight',          'fungal',    _FUNG,                                                       10.0),
    ('septoria',        'fungal',    _FUNG,                                                       10.0),
    ('gray leaf spot',  'fungal',    'Strobilurin fungicide',                                     10.0),
    ('grey leaf spot',  'fungal',    'Strobilurin fungicide',                                     10.0),
    ('scab',            'fungal',    'Captan / myclobutanil fungicide',                           10.0),
    ('black rot',       'fungal',    'Mancozeb / myclobutanil fungicide',                         10.0),
    ('target spot',     'fungal',    _FUNG,                                                       10.0),
    ('leaf scorch',     'fungal',    _FUNG,                                                       10.0),
    ('mold',            'fungal',    'Broad-spectrum fungicide',                                  10.0),
    ('mould',           'fungal',    'Broad-spectrum fungicide',                                  10.0),
    ('leaf spot',       'fungal',    _FUNG,                                                       10.0),
]

PLANT_WORDS = ['bell pepper', 'apple', 'blueberry', 'cherry', 'corn', 'grape', 'peach', 'potato',
               'raspberry', 'soybean', 'soyabean', 'squash', 'strawberry', 'tomato']

HEALTHY_LABEL = 'None (healthy)'
GROUPS = ('healthy', 'fungal', 'bacterial')


def norm(name):
    return re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()


def _match(name):
    n = f' {norm(name)} '
    for kw, grp, pesticide, ml in TREATMENTS:
        if f' {kw} ' in n:
            return grp, pesticide, ml
    return None


def group_for(class_name: str):
    m = _match(class_name)
    if m:
        return m[0]
    n = norm(class_name)
    if 'healthy' in n or re.fullmatch(r'[a-z0-9 ]+ leaf', n):
        return 'healthy'
    return ''


def plan_for(class_name: str):
    m = _match(class_name)
    return (m[1], m[2]) if m else (HEALTHY_LABEL, 0.0)


def split_class(class_name: str):
    """-> (plant label, group, disease text).
    'apple_fungal' -> ('Apple leaf', 'fungal', '');  'Tomato_Early_blight_leaf' ->
    ('Tomato leaf', 'fungal', 'early blight');  'Blueberry_leaf' -> ('Blueberry leaf', 'healthy', '')."""
    n = class_name.strip()
    if n.lower() in GROUPS:
        return 'Leaf', n.lower(), ''
    if '_' in n:
        plant, group = n.rsplit('_', 1)
        if group.lower() in GROUPS:
            return plant.replace('_', ' ').capitalize() + ' leaf', group.lower(), ''
    nn = norm(n)
    for p in PLANT_WORDS:
        if nn == p or nn.startswith(p + ' '):
            disease = re.sub(r'\bleaf\b', '', nn[len(p):]).strip()
            return p.capitalize() + ' leaf', group_for(n), disease
    return nn.capitalize(), group_for(n), ''


class Smoother:
    """Tiny IoU tracker: a detection is only reported after it has been seen in
    `min_hits` frames; its class is a confidence-weighted vote over the last
    `n_frames` sightings. Removes one-frame false alarms and label flicker."""

    def __init__(self, n_frames=5, min_hits=3, iou_thr=0.3, max_miss=2):
        self.n, self.min_hits, self.iou_thr, self.max_miss = n_frames, min_hits, iou_thr, max_miss
        self.tracks = []

    @staticmethod
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, dets):
        """dets: list of (x1, y1, x2, y2, class_id, conf). Returns confirmed ones."""
        for t in self.tracks:
            t['matched'] = False
        for d in sorted(dets, key=lambda d: -d[5]):
            best, best_iou = None, self.iou_thr
            for t in self.tracks:
                if t['matched']:
                    continue
                v = self.iou(t['box'], d[:4])
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:
                best = {'box': tuple(float(v) for v in d[:4]), 'hist': deque(maxlen=self.n),
                        'miss': 0, 'matched': False}
                self.tracks.append(best)
            else:   # smooth the box a little
                best['box'] = tuple(0.5 * o + 0.5 * float(v) for o, v in zip(best['box'], d[:4]))
            best['hist'].append((d[4], d[5]))
            best['miss'] = 0
            best['matched'] = True
        for t in self.tracks:
            if not t['matched']:
                t['miss'] += 1
        self.tracks = [t for t in self.tracks if t['miss'] <= self.max_miss]

        out = []
        for t in self.tracks:
            if t['miss'] == 0 and len(t['hist']) >= self.min_hits:
                score = Counter()
                for cid, conf in t['hist']:
                    score[cid] += conf
                cid = score.most_common(1)[0][0]
                confs = [c for k, c in t['hist'] if k == cid]
                x1, y1, x2, y2 = t['box']
                out.append((int(x1), int(y1), int(x2), int(y2), cid, sum(confs) / len(confs)))
        return out


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.declare_parameter('model_path',
                               os.path.expanduser('~/agrispray_ws/models/best.onnx'))
        self.declare_parameter('conf_threshold', 0.40)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('threads', 4)
        self.declare_parameter('pump_flow_ml_per_s', 10.0)  # CALIBRATE THIS
        self.declare_parameter('max_spray_seconds', 5.0)
        self.declare_parameter('ref_area_frac', 0.15)       # box area that = 1.0x dose
        self.declare_parameter('in_topic', '/camera/image/compressed')
        self.declare_parameter('stream_port', 8080)   # 0 = disable web stream
        # --- one specialist model per crop: <models_dir>/<crop>.onnx ---
        self.declare_parameter('models_dir', os.path.expanduser('~/agrispray_ws/models'))
        self.declare_parameter('crop', '')            # '' / 'auto' = use model_path (general model)
        # --- accuracy helpers ---
        self.declare_parameter('disease_conf_threshold', 0.65)  # stricter for red (disease) boxes
        self.declare_parameter('min_box_frac', 0.01)   # ignore boxes smaller than 1% of the frame
        self.declare_parameter('max_box_frac', 0.95)   # ignore boxes covering the whole frame
        self.declare_parameter('smooth_frames', 5)     # frames used for the class vote
        self.declare_parameter('min_hits', 3)          # sightings needed before a box is shown (1 = off)
        # --- classification-model support (model that outputs one label per image, no boxes) ---
        self.declare_parameter('names_file', '')       # classes.txt (one per line) or data.yaml
        self.declare_parameter('min_green_frac', 0.15) # classifier only: need this much leaf colour in view
        self.declare_parameter('cls_norm', 'unit')     # 'unit' (/255) or 'imagenet' (mean/std)
        # --- save frames + auto pre-labels for retraining (empty = off) ---
        self.declare_parameter('collect_dir', '')
        self.declare_parameter('collect_interval', 2.0)
        self.declare_parameter('collect_max', 1500)

        model_path = self.get_parameter('model_path').value
        self.conf_thr = float(self.get_parameter('conf_threshold').value)
        self.iou_thr = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.flow = float(self.get_parameter('pump_flow_ml_per_s').value)
        self.max_spray = float(self.get_parameter('max_spray_seconds').value)
        self.ref_area = float(self.get_parameter('ref_area_frac').value)
        self.disease_conf = float(self.get_parameter('disease_conf_threshold').value)
        self.min_box_frac = float(self.get_parameter('min_box_frac').value)
        self.max_box_frac = float(self.get_parameter('max_box_frac').value)
        self.smooth_n = int(self.get_parameter('smooth_frames').value)
        self.min_hits = int(self.get_parameter('min_hits').value)
        self.smoother = Smoother(self.smooth_n, self.min_hits)
        self.min_green_frac = float(self.get_parameter('min_green_frac').value)
        self.cls_norm = str(self.get_parameter('cls_norm').value).lower()
        self._green_frac = 0.0
        self.is_cls = False
        self.collect_dir = os.path.expanduser(str(self.get_parameter('collect_dir').value))
        self.collect_interval = float(self.get_parameter('collect_interval').value)
        self.collect_max = int(self.get_parameter('collect_max').value)
        self.collect_last = 0.0
        self.collect_n = 0
        if self.collect_dir:
            os.makedirs(self.collect_dir, exist_ok=True)

        self.model_path = model_path
        self.models_dir = os.path.expanduser(str(self.get_parameter('models_dir').value))
        self.crop = ''
        crop0 = str(self.get_parameter('crop').value).strip().lower()
        if crop0 and crop0 not in ('auto', 'general'):
            path = self._crop_path(crop0)
            if path is None:
                raise FileNotFoundError(
                    f'No model for crop "{crop0}": expected {self.models_dir}/{crop0}.onnx')
            self._load_model(path)
            self.crop = crop0
        else:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f'Model not found: {model_path}  (set -p model_path:=/path/best.onnx)')
            self._load_model(model_path)

        qos = qos_profile_sensor_data
        self.sub = self.create_subscription(
            CompressedImage, self.get_parameter('in_topic').value, self.on_image, qos)
        self.pub_img = self.create_publisher(
            CompressedImage, '/detector/image_annotated/compressed', qos)
        self.pub_det = self.create_publisher(String, '/detections', 10)
        self.create_subscription(String, '/set_crop', self.on_set_crop, 10)

        self.n_frames = 0
        self.n_rx = 0
        self.t_sum = 0.0
        self.cond = threading.Condition()
        self.last_jpg = None
        port = int(self.get_parameter('stream_port').value)
        if port > 0:
            self._start_stream(port)
        self.create_timer(5.0, self._stats)

        self.get_logger().info(
            f'Model loaded: {self.active_model} (crop={self.crop or "general"}) | input {self.imgsz}px | {len(self.names)} classes')

    # ------------------------------------------------------------------
    def _stats(self):
        if self.n_frames == 0 and self.n_rx > 0:
            self.get_logger().warn(
                f'Detector: received {self.n_rx} camera frames but inference FAILS on all of '
                'them - see the "Inference failed" error above (usually the wrong ONNX model)')
        elif self.n_frames == 0:
            self.get_logger().warn('Detector: no camera frames received yet')
        else:
            self.get_logger().info(
                f'Detector alive: {self.n_frames / 5.0:.1f} fps, '
                f'{self.t_sum / self.n_frames:.0f} ms/inference')
        self.n_frames, self.n_rx, self.t_sum = 0, 0, 0.0

    def _start_stream(self, port):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == '/':
                    body = (b'<html><body style="margin:0;background:#111">'
                            b'<img src="/stream" style="width:100%"></body></html>')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != '/stream':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        with outer.cond:
                            outer.cond.wait(timeout=2.0)
                            jpg = outer.last_jpg
                        if jpg is None:
                            continue
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                            + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.get_logger().info(f'Live view: http://<pi-ip>:{port}')

    def _crop_path(self, crop):
        if not self.models_dir:
            return None
        p = os.path.join(self.models_dir, f'{crop}.onnx')
        return p if os.path.isfile(p) else None

    def _load_model(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(self.get_parameter('threads').value)
        session = ort.InferenceSession(
            path, sess_options=so, providers=['CPUExecutionProvider'])
        inp = session.get_inputs()[0]
        shp = inp.shape  # e.g. [1,3,416,416]
        imgsz = (int(shp[2]) if isinstance(shp[2], int) and isinstance(shp[3], int)
                 else int(self.get_parameter('imgsz').value))
        self.session, self.input_name, self.imgsz = session, inp.name, imgsz
        self.active_model = path
        self.names = self._load_names()
        self.get_logger().info(
            f'ONNX inputs : {[(i.name, i.shape) for i in self.session.get_inputs()]}')
        self.get_logger().info(
            f'ONNX outputs: {[(o.name, o.shape) for o in self.session.get_outputs()]}')
        oshape = self.session.get_outputs()[0].shape
        self.is_cls = (len(oshape) == 2)          # detector outputs are 3-D, classifier is (1, n_classes)
        if self.is_cls:
            nc = oshape[1] if isinstance(oshape[1], int) else None
            self.get_logger().info(
                'CLASSIFICATION model detected: one label per frame (centre square of the image, '
                'no boxes). Robot decisions use that single label.')
            if nc is not None and len(self.names) != nc:
                self.get_logger().error(
                    f'The classifier has {nc} outputs but {len(self.names)} class names were found. '
                    'Using generic names class_0.. so NOTHING gets sprayed. Give the real names '
                    'with  -p names_file:=/path/classes.txt  (one per line, in the model class order).')
                self.names = [f'class_{i}' for i in range(nc)]

    def on_set_crop(self, msg):
        """ros2 topic pub --once /set_crop std_msgs/msg/String "{data: tomato}"
        ('auto' goes back to the general model)."""
        crop = msg.data.strip().lower()
        if crop in ('', 'auto', 'general'):
            path, new_crop = self.model_path, ''
        else:
            path, new_crop = self._crop_path(crop), crop
            if path is None:
                self.get_logger().error(
                    f'No model for crop "{crop}" (expected {self.models_dir}/{crop}.onnx) '
                    '- keeping the current model')
                return
        try:
            self._load_model(path)
        except Exception as e:  # noqa
            self.get_logger().error(f'Could not load {path}: {e} - keeping the current model')
            return
        self.crop = new_crop
        self.smoother = Smoother(self.smooth_n, self.min_hits)   # forget old tracks
        self.get_logger().info(
            f'Switched model -> {path} (crop={self.crop or "general"}), {len(self.names)} classes')

    def _load_names(self):
        nf = os.path.expanduser(str(self.get_parameter('names_file').value))
        if nf:
            try:
                if nf.lower().endswith(('.yaml', '.yml')):
                    import yaml
                    d = yaml.safe_load(open(nf))
                    names = d['names'] if isinstance(d, dict) else d
                    names = ([names[k] for k in sorted(names)] if isinstance(names, dict)
                             else list(names))
                else:
                    names = [l.strip() for l in open(nf).read().splitlines() if l.strip()]
                self.get_logger().info(f'Class names read from {nf} ({len(names)} classes).')
                return names
            except Exception as e:  # noqa
                self.get_logger().error(f'Could not read names_file {nf}: {e}')
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if 'names' in meta:
                d = ast.literal_eval(meta['names'])
                names = [d[k] for k in sorted(d)]
                self.get_logger().info('Class names read from ONNX metadata.')
                return names
        except Exception as e:  # noqa
            self.get_logger().warn(f'Could not read names from ONNX: {e}')
        self.get_logger().warn('Using DEFAULT_NAMES - verify them against data.yaml!')
        return list(DEFAULT_NAMES)

    def _name(self, cid):
        return self.names[cid] if 0 <= cid < len(self.names) else f'class_{cid}'

    # ------------------------------------------------------------------
    def _raw_classify(self, frame):
        """Classification model: classify the centre square of the frame. Returns one
        (x1,y1,x2,y2,class_id,conf) tuple covering that square, or [] if no leaf-like
        colours are in view (a classifier always names *something*, even a wall)."""
        H, W = frame.shape[:2]
        side = min(H, W)
        x0, y0 = (W - side) // 2, (H - side) // 2
        crop = frame[y0:y0 + side, x0:x0 + side]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        leafish = ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 95)
                   & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40))
        self._green_frac = float(leafish.mean())
        if self._green_frac < self.min_green_frac:
            return []
        rgb = cv2.cvtColor(cv2.resize(crop, (self.imgsz, self.imgsz),
                                      interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.cls_norm == 'imagenet':
            rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
                  np.array([0.229, 0.224, 0.225], np.float32)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        out = np.asarray(self.session.run(None, {self.input_name: blob})[0],
                         dtype=np.float32).reshape(-1)
        if out.min() < 0 or out.max() > 1 or abs(float(out.sum()) - 1.0) > 0.01:
            e = np.exp(out - out.max())           # logits -> probabilities
            out = e / e.sum()
        cid = int(out.argmax())
        conf = float(out[cid])
        if conf < self.conf_thr:
            return []
        if plan_for(self._name(cid))[1] > 0 and conf < self.disease_conf:
            return []                             # disease needs higher confidence
        return [(x0, y0, x0 + side, y0 + side, cid, conf)]

    def _raw_detect(self, frame):
        """ONNX inference + NMS + sanity filters. Returns (x1,y1,x2,y2,class_id,conf) tuples."""
        if self.is_cls:
            return self._raw_classify(frame)
        H, W = frame.shape[:2]
        canvas, r, left, top = letterbox(frame, self.imgsz)
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        outs = self.session.run(None, {self.input_name: blob})
        out = np.asarray(outs[0])
        if out.ndim == 3:
            out = out[0]                          # drop the batch dimension
        if out.ndim != 2:
            raise ValueError(f'Unexpected model output shape {out.shape} '
                             f'(all outputs: {[np.asarray(o).shape for o in outs]})')
        nc = len(self.names)
        e2e = False
        if out.shape[0] == 4 + nc:                # (4+nc, N) -> (N, 4+nc)   [standard YOLOv8]
            out = out.T
        elif out.shape[1] == 4 + nc:              # already (N, 4+nc)
            pass
        elif out.shape[1] == 6:                   # NMS-free / nms=True export: x1,y1,x2,y2,conf,cls
            e2e = True
        elif out.shape[0] < out.shape[1]:         # fallback guess
            out = out.T

        if e2e:
            out = out[out[:, 4] >= self.conf_thr]
            if len(out) == 0:
                return []
            confs = out[:, 4]
            cls_ids = out[:, 5].astype(int)
            x = (out[:, 0] - left) / r
            y = (out[:, 1] - top) / r
            w = (out[:, 2] - out[:, 0]) / r
            h = (out[:, 3] - out[:, 1]) / r
        else:
            scores_all = out[:, 4:]
            cls_ids = scores_all.argmax(axis=1)
            confs = scores_all.max(axis=1)
            keep = confs >= self.conf_thr
            out, cls_ids, confs = out[keep], cls_ids[keep], confs[keep]
            if len(out) == 0:
                return []
            cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            x = (cx - bw / 2 - left) / r
            y = (cy - bh / 2 - top) / r
            w = bw / r
            h = bh / r
        boxes = np.stack([x, y, w, h], axis=1)

        idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(),
                               self.conf_thr, self.iou_thr)
        idx = np.array(idx).flatten() if len(idx) else []

        raw = []
        for i in idx:
            x1 = int(max(0, boxes[i][0]))
            y1 = int(max(0, boxes[i][1]))
            x2 = int(min(W - 1, boxes[i][0] + boxes[i][2]))
            y2 = int(min(H - 1, boxes[i][1] + boxes[i][3]))
            if x2 <= x1 or y2 <= y1:
                continue
            frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if frac < self.min_box_frac or frac > self.max_box_frac:
                continue                      # tiny specks / whole-frame boxes
            cid, conf = int(cls_ids[i]), float(confs[i])
            if plan_for(self._name(cid))[1] > 0 and conf < self.disease_conf:
                continue                      # disease needs higher confidence
            raw.append((x1, y1, x2, y2, cid, conf))
        return raw

    def infer(self, frame):
        H, W = frame.shape[:2]
        stable = self.smoother.update(self._raw_detect(frame))

        dets = []
        for (x1, y1, x2, y2, cid, conf) in stable:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            name = self._name(cid)
            plant_name, group, disease = split_class(name)
            if self.crop and plant_name == 'Leaf':
                plant_name = self.crop.replace('_', ' ').capitalize() + ' leaf'
            pesticide, base_ml = plan_for(name)
            area_frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if self.is_cls:       # no real box: use how much of the view is leaf-coloured
                area_frac *= self._green_frac

            dose_ml, spray_s = 0.0, 0.0
            if base_ml > 0:
                # bigger plant/leaf in view => more dose (0.5x .. 2.0x)
                scale = min(2.0, max(0.5, area_frac / self.ref_area))
                dose_ml = round(base_ml * scale, 1)
                spray_s = round(min(self.max_spray, dose_ml / self.flow), 2)

            dets.append({
                'class_id': int(cid),
                'plant': plant_name,
                'group': group,
                'disease': disease,
                'class_name': name,
                'conf': round(float(conf), 3),
                'bbox': [x1, y1, x2, y2],
                'area_frac': round(area_frac, 4),
                'cx_frac': round(((x1 + x2) / 2) / W, 3),
                'pesticide': pesticide,
                'dose_ml': dose_ml,
                'spray_seconds': spray_s,
            })
        return dets

    def _collect(self, frame, dets):
        """Save the raw frame + YOLO-format pre-labels (to be corrected in Roboflow)."""
        now = time.monotonic()
        if (not self.collect_dir or self.collect_n >= self.collect_max
                or now - self.collect_last < self.collect_interval):
            return
        self.collect_last = now
        H, W = frame.shape[:2]
        stem = os.path.join(self.collect_dir, f'frame_{int(time.time() * 1000)}')
        cv2.imwrite(stem + '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(stem + '.txt', 'w') as f:
            for d in dets:
                x1, y1, x2, y2 = d['bbox']
                f.write(f"{d['class_id']} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        self.collect_n += 1

    # ------------------------------------------------------------------
    def draw(self, frame, dets):
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            sprayable = d['spray_seconds'] > 0
            color = ((0, 0, 255) if sprayable else
                     (0, 165, 255) if d['group'] in ('viral', 'pest') else (0, 200, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['plant']} ({d['conf']:.2f}) {d['group']}".strip()
            if d.get('disease'):
                label += f" - {d['disease']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(frame, label, (x1 + 2, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if sprayable:
                info = f"{d['pesticide'][:28]} | {d['dose_ml']}ml {d['spray_seconds']}s"
                cv2.putText(frame, info, (x1, min(frame.shape[0] - 5, y2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame

    def on_image(self, msg: CompressedImage):
        self.n_rx += 1
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        t0 = time.monotonic()
        try:
            dets = self.infer(frame)
        except Exception as e:  # noqa
            self.get_logger().error(
                f'Inference failed: {e!r}\n{traceback.format_exc()}',
                throttle_duration_sec=5.0)
            # still show the plain camera image so the web page is not black
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.cond:
                    self.last_jpg = jpg.tobytes()
                    self.cond.notify_all()
            return
        dt = (time.monotonic() - t0) * 1000.0
        self.n_frames += 1
        self.t_sum += dt

        # detections for the controller (published every frame, even if empty)
        out = {
            'stamp': time.time(),
            'frame_w': frame.shape[1],
            'frame_h': frame.shape[0],
            'detections': dets,
        }
        self.pub_det.publish(String(data=json.dumps(out)))

        self._collect(frame, dets)          # must run before draw() paints on the frame

        # annotated image
        annotated = self.draw(frame, dets)
        ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.last_jpg = jpg.tobytes()
                self.cond.notify_all()
            m = CompressedImage()
            m.header = msg.header
            m.format = 'jpeg'
            m.data = jpg.tobytes()
            self.pub_img.publish(m)

        if dets:
            self.get_logger().info(
                f'{len(dets)} detection(s), {dt:.0f} ms: ' +
                ', '.join(f"{d['plant']}({d['conf']:.2f})" for d in dets),
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
