import os
import time

import cv2
import numpy as np

import config
from utils import LandmarkExtractor, preprocess_frame, extract_keypoints, draw_landmarks


KATA_YANG_DIREKAM = ["tidak"]
JEDA_DETIK = 2.0  


def next_index(word):
    folder = os.path.join(config.DATA_PATH, word)
    os.makedirs(folder, exist_ok=True)
    nums = [int(f[:-4]) for f in os.listdir(folder) if f.endswith(".npy") and f[:-4].isdigit()]
    return max(nums, default=-1) + 1


def put_text(frame, text, pos, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    words = KATA_YANG_DIREKAM or config.ACTIONS
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    stop = False

    for word in words:
        if stop:
            break
        start = next_index(word)
        print(f"\n[{word}] sudah ada {start} rekaman, target {config.NUM_SEQUENCES}.")

        for seq in range(start, config.NUM_SEQUENCES):
            t_end = time.time() + JEDA_DETIK
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    stop = True
                    break
                frame = preprocess_frame(frame)
                hand_res, pose_res = extractor.process(frame)
                draw_landmarks(frame, hand_res, pose_res)
                sisa = t_end - time.time()
                put_text(frame, f"Kata: {config.DISPLAY_TEXT[word] or word}", (10, 35), 1.0, (0, 255, 255))
                put_text(frame, f"Rekaman {seq + 1}/{config.NUM_SEQUENCES}", (10, 70))
                put_text(frame, f"BERSIAP... {sisa:.1f}", (10, 110), 1.0, (0, 165, 255))
                cv2.imshow("Rekam Dataset BISINDO", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break
            if stop:
                break

            sequence = []
            while len(sequence) < config.SEQUENCE_LENGTH:
                ok, frame = cap.read()
                if not ok:
                    stop = True
                    break
                frame = preprocess_frame(frame)
                hand_res, pose_res = extractor.process(frame)
                sequence.append(extract_keypoints(hand_res, pose_res))
                draw_landmarks(frame, hand_res, pose_res)
                put_text(frame, f"Kata: {config.DISPLAY_TEXT[word] or word}", (10, 35), 1.0, (0, 255, 255))
                put_text(frame, f"MEREKAM {len(sequence)}/{config.SEQUENCE_LENGTH}", (10, 110), 1.0, (0, 0, 255))
                cv2.imshow("Rekam Dataset BISINDO", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break
            if stop:
                break

            path = os.path.join(config.DATA_PATH, word, f"{seq}.npy")
            np.save(path, np.array(sequence, dtype=np.float32))
            print(f"  tersimpan: {path}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\nSelesai merekam.")


if __name__ == "__main__":
    main()