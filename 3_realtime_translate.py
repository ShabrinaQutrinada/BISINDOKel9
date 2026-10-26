import json
import math
import os
import time
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

import config
from utils import (LandmarkExtractor, preprocess_frame, extract_keypoints,
                   draw_landmarks, hands_detected)

CONFIDENCE_THRESHOLD = 0.80
STABLE_WINDOW = 10
STABLE_MIN = 8
COOLDOWN_FRAMES = 20
PREDICT_EVERY = 2
MAX_HISTORY = 10
MAX_CAPTION = 4

WINDOW_NAME = "BISINDO Sign to Text"
CANVAS_W, CANVAS_H = 1280, 720
CAM_X, CAM_Y, CAM_W, CAM_H = 24, 80, 800, 600
SIDE_X, SIDE_W = 848, 408
ANIM_FRAMES = 22

BG_TOP = (17, 24, 39)
BG_BOTTOM = (3, 7, 18)
CARD = (30, 41, 59)
CARD_LINE = (51, 65, 85)
ACCENT = (45, 212, 191)
ACCENT_DARK = (17, 94, 89)
VIOLET = (167, 139, 250)
WHITE = (241, 245, 249)
MUTED = (148, 163, 184)
DIM = (100, 116, 139)

CARD1 = (SIDE_X, 80, SIDE_X + SIDE_W, 268)
CARD2 = (SIDE_X, 284, SIDE_X + SIDE_W, 680)
HEADER_RIGHT_X = 880


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def blend(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    win = "C:/Windows/Fonts/"
    paths = ([win + "segoeuib.ttf", win + "arialbd.ttf"] if bold
             else [win + "segoeui.ttf", win + "arial.ttf"])
    try:
        import matplotlib
        ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        paths.append(os.path.join(ttf, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    except ImportError:
        pass
    f = None
    for p in paths:
        if os.path.exists(p):
            f = ImageFont.truetype(p, size)
            break
    _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


class UI:
    def __init__(self):
        self.base_rgb = self._build_base()
        self.base = cv2.cvtColor(self.base_rgb, cv2.COLOR_RGB2BGR)

        m = Image.new("L", (CAM_W, CAM_H), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, CAM_W - 1, CAM_H - 1), radius=22, fill=255)
        ys, xs = np.where(np.asarray(m) == 0)
        self.corner = (ys + CAM_Y, xs + CAM_X)

        pad = 4
        ring = Image.new("L", (CAM_W + 2 * pad + 1, CAM_H + 2 * pad + 1), 0)
        ImageDraw.Draw(ring).rounded_rectangle(
            (pad - 2, pad - 2, CAM_W + pad + 2, CAM_H + pad + 2), radius=24, outline=255, width=3)
        ys, xs = np.where(np.asarray(ring) > 0)
        self.ring = (ys + CAM_Y - pad, xs + CAM_X - pad)

        sh = 150
        alpha = np.linspace(0, 0.85, sh) ** 1.4
        self.shade_h = sh
        self.shade = (1 - alpha)[:, None, None].astype(np.float32)

        self.live_patch = self._text_patch(
            lambda d: (d.rounded_rectangle((0, 0, 68, 26), radius=13, fill=(15, 23, 42, 255)),
                       d.text((24, 3), "LIVE", font=font(15, True), fill=WHITE + (255,))),
            70, 28)
        self.prep_patch = self._text_patch(
            lambda d: d.text((0, 0), "Menyiapkan...", font=font(14), fill=WHITE + (255,)), 120, 20)

        self.cache = {}

    def _build_base(self):
        grad = np.linspace(0, 1, CANVAS_H)[:, None, None]
        top = np.array(BG_TOP, dtype=np.float32)
        bottom = np.array(BG_BOTTOM, dtype=np.float32)
        base = (top * (1 - grad) + bottom * grad).repeat(CANVAS_W, axis=1).astype(np.uint8)
        img = Image.fromarray(base)
        d = ImageDraw.Draw(img)
        for box in (CARD1, CARD2):
            d.rounded_rectangle(box, radius=18, fill=CARD, outline=CARD_LINE, width=1)
        d.rounded_rectangle((CAM_X, CAM_Y, CAM_X + CAM_W, CAM_Y + CAM_H), radius=22, fill=CARD)
        d.text((CARD1[0] + 22, CARD1[1] + 18), "ISYARAT TERDETEKSI", font=font(14, True), fill=MUTED)
        d.text((CARD2[0] + 22, CARD2[1] + 18), "RIWAYAT", font=font(14, True), fill=MUTED)
        d.text((CARD1[0] + 22, CARD1[3] - 56), "Kestabilan isyarat", font=font(13), fill=DIM)
        d.rounded_rectangle((24, 18, 136, 52), radius=10, fill=ACCENT)
        d.text((36, 22), "BISINDO", font=font(20, True), fill=(4, 47, 46))
        d.text((150, 21), "Sign Language to Text", font=font(22, True), fill=WHITE)
        hint_y = CAM_Y + CAM_H + 12
        x = CAM_X
        for key, label in [("Q", "keluar"), ("C", "hapus riwayat")]:
            d.rounded_rectangle((x, hint_y, x + 26, hint_y + 22), radius=6, fill=CARD, outline=CARD_LINE)
            d.text((x + 8, hint_y + 2), key, font=font(14, True), fill=WHITE)
            d.text((x + 34, hint_y + 2), label, font=font(14), fill=MUTED)
            x += 34 + int(font(14).getlength(label)) + 24
        credit = "MediaPipe  +  LSTM"
        d.text((CAM_X + CAM_W - font(14).getlength(credit), hint_y + 2), credit, font=font(14), fill=DIM)
        return np.asarray(img).copy()

    @staticmethod
    def _text_patch(draw_fn, w, h):
        """Gambar transparan (RGBA) -> (warna BGR, alpha) untuk ditempel di atas kamera."""
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(img))
        arr = np.asarray(img).astype(np.float32)
        color = arr[:, :, 2::-1].copy()
        alpha = arr[:, :, 3:4] / 255.0
        return color, alpha

    @staticmethod
    def _paste_alpha(canvas, patch, x, y):
        color, alpha = patch
        h, w = alpha.shape[:2]
        roi = canvas[y:y + h, x:x + w].astype(np.float32)
        canvas[y:y + h, x:x + w] = (roi * (1 - alpha) + color * alpha).astype(np.uint8)

    def _crop_draw(self, box):
        x1, y1, x2, y2 = box
        img = Image.fromarray(self.base_rgb[y1:y2, x1:x2].copy())
        return img, ImageDraw.Draw(img)

    def _cached(self, name, key, builder):
        hit = self.cache.get(name)
        if hit is None or hit[0] != key:
            hit = (key, builder())
            self.cache[name] = hit
        return hit[1]

    def _caption_patch(self, caption):
        f_cap = font(34, True)
        sep = "  \u00b7  "

        def draw(d):
            d.text((0, 0), "TERJEMAHAN", font=font(13, True), fill=MUTED + (255,))
            if not caption:
                d.text((0, 30), "Lakukan isyarat di depan kamera", font=font(28), fill=MUTED + (255,))
                return
            x = 0
            for i, w in enumerate(caption):
                last = i == len(caption) - 1
                d.text((x, 26), w, font=f_cap, fill=(ACCENT if last else WHITE) + (255,))
                x += f_cap.getlength(w)
                if not last:
                    d.text((x, 26), sep, font=f_cap, fill=MUTED + (255,))
                    x += f_cap.getlength(sep)

        return self._text_patch(draw, CAM_W - 56, 72)

    def _header_right(self, hand, fps):
        box = (HEADER_RIGHT_X, 0, CANVAS_W, 70)
        img, d = self._crop_draw(box)
        ox = box[0]
        status = "Tangan terdeteksi" if hand else "Menunggu tangan"
        f_st = font(16, True)
        fps_txt = f"{fps} FPS"
        fps_w = font(14).getlength(fps_txt)
        pill_w = int(46 + f_st.getlength(status))
        px2 = CANVAS_W - 24 - fps_w - 18 - ox
        px1 = px2 - pill_w
        d.rounded_rectangle((px1, 18, px2, 52), radius=17, fill=CARD, outline=CARD_LINE)
        d.ellipse((px1 + 14, 29, px1 + 26, 41), fill=ACCENT if hand else DIM)
        d.text((px1 + 34, 25), status, font=f_st, fill=WHITE if hand else MUTED)
        d.text((CANVAS_W - 24 - fps_w - ox, 27), fps_txt, font=font(14), fill=DIM)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _side_panel(self, current, history, stability, anim):
        box = (SIDE_X, 80, SIDE_X + SIDE_W + 1, 681)
        img, d = self._crop_draw(box)
        ox, oy = box[0], box[1]

        cx = SIDE_W // 2
        top1 = CARD1[1] - oy
        if current:
            t = 1 - anim / ANIM_FRAMES if anim > 0 else 1.0
            size = int(58 + 16 * (1 - t) ** 2)
            f_big = font(size, True)
            wbig = f_big.getlength(current)
            if wbig > SIDE_W - 40:
                size = int(size * (SIDE_W - 40) / wbig)
                f_big = font(size, True)
                wbig = f_big.getlength(current)
            d.text((cx - wbig / 2, top1 + 62 - (size - 58) / 2), current, font=f_big,
                   fill=blend(ACCENT, WHITE, t * 0.35))
        else:
            msg = "Menunggu isyarat"
            d.text((cx - font(26).getlength(msg) / 2, top1 + 78), msg, font=font(26), fill=DIM)

        bx1, bx2, by = 22, SIDE_W - 22, CARD1[3] - 34 - oy
        d.rounded_rectangle((bx1, by, bx2, by + 8), radius=4, fill=(15, 23, 42))
        fill_w = int((bx2 - bx1) * stability)
        if fill_w > 8:
            d.rounded_rectangle((bx1, by, bx1 + fill_w, by + 8), radius=4, fill=VIOLET)

        y = CARD2[1] - oy + 48
        if not history:
            d.text((22, y), "Belum ada kata", font=font(16), fill=DIM)
        for i, w in enumerate(reversed(history)):
            newest = i == 0
            d.ellipse((22, y + 6, 32, y + 16), fill=ACCENT if newest else CARD_LINE)
            d.text((44, y), w, font=font(18, newest), fill=WHITE if newest else MUTED)
            if newest:
                tag = "baru"
                tw = font(12, True).getlength(tag)
                tx = 52 + font(18, True).getlength(w)
                d.rounded_rectangle((tx, y + 2, tx + tw + 14, y + 22), radius=10, fill=ACCENT_DARK)
                d.text((tx + 7, y + 4), tag, font=font(12, True), fill=ACCENT)
            y += 34
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def render(self, cam_bgr, state):
        canvas = self.base.copy()
        t_now = state["time"]
        hand = state["hand"]
        pulse = 0.5 + 0.5 * math.sin(t_now * 4)

        cam = cam_bgr.copy()
        cam[-self.shade_h:] = (cam[-self.shade_h:].astype(np.float32) * self.shade).astype(np.uint8)
        canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam
        canvas[self.corner] = self.base[self.corner]

        ring_color = blend(ACCENT_DARK, ACCENT, pulse) if hand else CARD_LINE
        canvas[self.ring] = bgr(ring_color)

        self._paste_alpha(canvas, self.live_patch, CAM_X + 16, CAM_Y + 16)
        r = int(4 + 1.5 * pulse)
        cv2.circle(canvas, (CAM_X + 29, CAM_Y + 29), r, (113, 113, 248), -1, cv2.LINE_AA)

        if state["buffer_ratio"] < 1.0:
            x1, x2 = CAM_X + 20, CAM_X + CAM_W - 20
            cv2.rectangle(canvas, (x1, CAM_Y + 56), (x2, CAM_Y + 62), bgr(CARD), -1)
            cv2.rectangle(canvas, (x1, CAM_Y + 56), (x1 + int((x2 - x1) * state["buffer_ratio"]), CAM_Y + 62),
                          bgr(ACCENT), -1)
            self._paste_alpha(canvas, self.prep_patch, CAM_X + 20, CAM_Y + 68)

        caption = list(state["history"][-MAX_CAPTION:])
        f_cap = font(34, True)
        while len(caption) > 1 and f_cap.getlength("  \u00b7  ".join(caption)) > CAM_W - 56:
            caption.pop(0)
        cap_patch = self._cached("caption", tuple(caption), lambda: self._caption_patch(caption))
        self._paste_alpha(canvas, cap_patch, CAM_X + 28, CAM_Y + CAM_H - 96)

        header = self._cached("header", (hand, state["fps"]),
                              lambda: self._header_right(hand, state["fps"]))
        canvas[0:70, HEADER_RIGHT_X:CANVAS_W] = header

        key = (state["current"], tuple(state["history"]), round(state["stability"], 1), state["anim"])
        side = self._cached("side", key, lambda: self._side_panel(
            state["current"], state["history"], round(state["stability"], 1), state["anim"]))
        canvas[80:681, SIDE_X:SIDE_X + SIDE_W + 1] = side

        return canvas


def open_camera(index):
    """Membuka webcam dengan pengaturan cepat (DirectShow + MJPG di Windows)."""
    cap = None
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.LABELS_PATH) as f:
        labels = json.load(f)
    print("Kelas:", labels)

    infer = tf.function(lambda x: model(x, training=False))

    extractor = LandmarkExtractor()
    cap = open_camera(config.CAMERA_INDEX)
    ui = UI()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    sequence = deque(maxlen=config.SEQUENCE_LENGTH)
    recent = deque(maxlen=STABLE_WINDOW)
    history = []
    frame_no = 0
    cooldown = 0
    current = ""
    anim = 0
    stability = 0.0
    frame_times = deque(maxlen=30)
    fps_shown = 0
    fps_updated = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam tidak terbaca.")
            break
        frame_no += 1

        frame = preprocess_frame(frame)
        hand_res, pose_res = extractor.process(frame)
        sequence.append(extract_keypoints(hand_res, pose_res))
        hand_now = hands_detected(hand_res)

        if cooldown > 0:
            cooldown -= 1
        if anim > 0:
            anim -= 1

        if len(sequence) == config.SEQUENCE_LENGTH and frame_no % PREDICT_EVERY == 0:
            if hand_now:
                x = tf.constant(np.expand_dims(np.array(sequence, dtype=np.float32), axis=0))
                probs = infer(x).numpy()[0]
                best = int(np.argmax(probs))
                recent.append(best if probs[best] >= CONFIDENCE_THRESHOLD else -1)
            else:
                recent.append(-1)

            idx, count = Counter(recent).most_common(1)[0]
            stability = count / STABLE_WINDOW if idx != -1 else 0.0

            if len(recent) == STABLE_WINDOW:
                if idx != -1 and count >= STABLE_MIN:
                    word = labels[idx]
                    new_text = config.DISPLAY_TEXT.get(word, word)
                    if new_text and new_text != current:
                        anim = ANIM_FRAMES
                    current = new_text
                    if current and cooldown == 0 and (not history or history[-1] != current):
                        history.append(current)
                        history = history[-MAX_HISTORY:]
                        cooldown = COOLDOWN_FRAMES
                        anim = ANIM_FRAMES
                        recent.clear()
                elif idx == -1:
                    current = ""

        now = time.time()
        frame_times.append(now)
        if len(frame_times) > 1 and now - fps_updated > 0.5:
            fps_shown = int(round((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])))
            fps_updated = now

        cam = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        draw_landmarks(cam, hand_res, pose_res)
        canvas = ui.render(cam, {
            "history": history,
            "current": current,
            "hand": hand_now,
            "buffer_ratio": len(sequence) / config.SEQUENCE_LENGTH,
            "stability": stability,
            "fps": fps_shown,
            "anim": anim,
            "time": now,
        })

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            history = []

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()