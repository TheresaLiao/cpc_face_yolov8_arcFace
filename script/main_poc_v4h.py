import cv2
import time
import numpy as np
import glob
import json
import os
import torch
import datetime
from utils_poc import FaceManage, CodeBase


class CustomFaceManage(FaceManage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_reference_embeddings(self, imgDBpath):
        if imgDBpath is None:
            print("prepare_reference_embeddings: reference path is None")
            return 0
        if not os.path.isdir(imgDBpath):
            print(f"prepare_reference_embeddings: reference path not found: {imgDBpath}")
            return 0

        imgPaths_NeedInference, imgs = self.findNeedEmbeddingImgs(imgDBpath)
        if not imgPaths_NeedInference:
            print(f"No new images found for embedding in {imgDBpath}")
            return 0

        os.makedirs(os.path.join(CodeBase, "aligned"), exist_ok=True)

        for img, imgPath in zip(imgs, imgPaths_NeedInference):
            filename = os.path.basename(imgPath)
            if img is None:
                print(f"Cannot load image: {imgPath}")
                continue

            preds = self.inference([img])
            inference_informations = self.postprocess(preds)
            crops, _ = self.get_aligned_crops_max([img], inference_informations, NoneAsBlack=False)

            if not crops:
                print(f"File cannot find face : {filename}")
                continue

            aligned_crop = crops[0]
            base, _ = os.path.splitext(filename)
            cv2.imwrite(os.path.join(CodeBase, "aligned", f"{base}full.png"), img)
            cv2.imwrite(os.path.join(CodeBase, "aligned", f"{base}_face_aligned.png"), aligned_crop)

            embed_arr = self.get_embedding_from_imgs([aligned_crop])
            if len(embed_arr) == 0:
                print(f"Embed generation failed: {filename}")
                continue

            self.saveJson([imgPath], [embed_arr[0]], verbose=True)
            print(f"Saved json for {imgPath}")

        return 0
        
    def findNeedEmbeddingImgs(self, reference_imgs_path):
        """
        尋找資料夾中尚未產生對應 .json 特徵檔的新圖片
        覆寫尋找圖片的函式，增加對大寫 .JPG, .PNG 等副檔名的支援
        """
        imgPaths = []
        # 在這裡加入大寫的副檔名
        imgForms = ["png", "jpg", "JPG", "PNG", "jpeg", "JPEG"]
        shouldNotContain = "face_aligned"
        
        for imgForm in imgForms:
            imgPaths_tmp = glob.glob(f"{reference_imgs_path}/*.{imgForm}")
            for imgPath in imgPaths_tmp:
                if shouldNotContain in imgPath:
                    continue
                imgPaths.append(imgPath)
                
        imgPaths_NeedInference = []
        imgs = []
        for imgPath in imgPaths:
            # 原本 utils.py 是用 imgPath[:-3]+"json"，這裡改用更安全的 os.path.splitext
            base, ext = os.path.splitext(imgPath)
            jsonPath = base + ".json"
            
            if not os.path.exists(jsonPath):
                img = cv2.imread(imgPath)
                if img is not None:
                    imgs.append(img)
                    imgPaths_NeedInference.append(imgPath)
                    
        return imgPaths_NeedInference, imgs

    def get_reference_embeds_from_folder_jsons(self, folder, get_name_from_filename=True, filtering_by_date=False):
        """
        從資料夾中讀取所有 JSON 快取檔，組合成 PyTorch Tensor 矩陣供比對使用
        覆寫原有的特徵讀取函式，以符合「代號_照片號碼.JPG」的命名規則
        """
        jsonFiles = glob.glob(os.path.join(folder, "*.json"))
        corresponding_persons = []
        reference_embeds = []

        # 若沒有 json 檔案，先嘗試產生缺失的特徵 json
        if not len(jsonFiles):
            print("找不到任何特徵值 JSON 檔案，開始產生缺失特徵 json ...")
            self.prepare_reference_embeddings(folder)
            jsonFiles = glob.glob(os.path.join(folder, "*.json"))
            if not len(jsonFiles):
                print("仍找不到特徵值 JSON 檔案，請檢查資料庫資料或模型是否正常。")
                exit()

        # 若存在部分圖片缺少 json，先產生缺失 json
        missing_img_paths, _ = self.findNeedEmbeddingImgs(folder)
        if len(missing_img_paths) > 0:
            print(f"發現 {len(missing_img_paths)} 張缺少 json 的圖片，開始補產生特徵 json ...")
            self.prepare_reference_embeddings(folder)
            jsonFiles = glob.glob(os.path.join(folder, "*.json"))

        for jsonFile in jsonFiles:
            info = json.load(open(jsonFile))
            name = info['filename']
            
            if get_name_from_filename:
                # 檔名格式為 test_0_1/P1_1.JPG，取出 P1 作為代號
                basename = os.path.basename(name)
                person_id = basename.split("_")[0] 
                name = person_id
                
            corresponding_persons.append(name)
            reference_embed = torch.from_numpy(np.array(info['embed']))
            reference_embeds.append(reference_embed.unsqueeze(0))
            
        reference_embeds = torch.cat(reference_embeds).float()
        return reference_embeds, corresponding_persons, jsonFiles

    def inference(self, imgs):
        """
        覆寫推論函式，將推論解析度提升至 1920，以利偵測天花板監視器的小人臉
        """
        return self.model_det(imgs, imgsz=1280, verbose=False)

    def postprocess(self, preds, getMaxFace=True, maxFaceArea=10):
        """
        覆寫後處理函式
        """
        inference_informations = []
        for pred in preds:
            # 確保 pred 具有 boxes 屬性
            if not hasattr(pred, 'boxes') or pred.boxes is None or len(pred.boxes.data) == 0:
                inference_informations.append([None, None])
                continue
                
            keypoints = pred.keypoints.data.cpu().numpy()
            boxes = pred.boxes.data.cpu().numpy()
            
            if getMaxFace:
                # 修正原本 utils.py 中面積計算的 bug (原本是 boxes[:,3]*boxes[:,1])
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                maxFaceIdx = np.argmax(areas)
                maxArea = np.max(areas)
                if maxArea <= maxFaceArea:
                    inference_informations.append([None, None])
                    continue
                inference_informations.append([boxes[maxFaceIdx], keypoints[maxFaceIdx]])
            else:
                pass
                
        return inference_informations


def format_mm_ss(seconds):
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_seconds(seconds):
    return f"{max(0, int(seconds)):02d}"


def main():
    video_path = 'item2video.mp4'
    video_in_path = 'item2video_in.mp4'
    video_out_path = 'item2video_out.mp4'
    db_path = 'test_0_1'
    # 設定影片起始時間 (秒)
    start_time_sec = 0.0
    
    print("初始化人臉辨識模型與資料庫...")
    face_manager = CustomFaceManage(
        deviceDetect="cuda:0", 
        deviceRecog="cuda:0",
        referencePath=db_path,
        voting=False 
    )
    print("初始化完成！開始處理影片...")
    
    # 生成基於當前時間的日誌檔案名稱
    log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    
    # 創建檔案並寫入標題行
    with open(log_filename, "w") as f:
        f.write("id,entry_time,departure_time,stay_duration\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
        
    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
        print(f"已跳轉至影片 {start_time_sec} 秒處開始處理。")
    
    last_seen_time = {}
    disappear_threshold = 3.0 # 消失 3 秒以上重新紀錄
    similarity_threshold = 0.35 # 信心度門檻
    
    # 針對 unknown 的最小人臉面積過濾參數
    min_unknown_face_area = 5000 
    
    # --- 連續幀數過濾機制參數 ---
    min_consecutive_known = 5    # 已知人員需要連續出現的幀數
    min_consecutive_unknown = 32 # unknown 需要連續出現的幀數
    consecutive_counts = {}      # 記錄每個代號目前連續出現的幀數 (字典格式: {'P1': 5, 'unknown': 2})
    
    # --- 進出時間紀錄 ---
    entry_times = {}             # 記錄每個代號的 entry_time
    # ----------------------------------
    
    # 定義容易誤判的 ROI 區域頂點
    roi_pts = np.array([[1200, 350], [1250, 350], [1250, 400], [1200, 400]], np.int32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("影片播放結束。")
            break
            
        # 將容易誤判的 ROI 區域塗成白色以免被偵測
        cv2.fillPoly(frame, [roi_pts], (255, 255, 255))
            
        # 取得當前影片時間 (秒)
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 初始化當前 frame 的變數
        current_frame_identities = set() # 記錄這個 frame 偵測到的所有代號
        identitys = []
        similaritys = []
        valid_boxes = []
        
        # 1. 進行人臉偵測
        preds_list = face_manager.inference([frame])#[:-1]
        
        if len(preds_list) > 0:
            preds = preds_list[0]
            boxes = preds.boxes.data.cpu().numpy() if preds.boxes is not None else []
            keypoints = preds.keypoints.data.cpu().numpy() if preds.keypoints is not None else []

            if len(boxes) > 0:
                crops = []
                temp_valid_boxes = [] 
                
                # 2. 處理畫面中的「每一個」人臉
                for i in range(len(boxes)):
                    box = boxes[i]
                    lmk = keypoints[i][..., :-1] 
                    
                    crop = face_manager.align(frame, lmk)
                    crops.append(crop)
                    temp_valid_boxes.append(box)
                    
                if len(crops) > 0:
                    # 3. 取得特徵向量
                    embeds = face_manager.get_embedding_from_imgs(crops)
                    
                    # 4. 進行辨識比對
                    raw_identitys, raw_similaritys = face_manager.recognize(embeds, similarityThresHold=similarity_threshold)
                    
                    # 過濾掉面積太小的 unknown 人臉
                    for identity, sim, box in zip(raw_identitys, raw_similaritys, temp_valid_boxes):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        
                        if identity == 'unknown' and area < min_unknown_face_area:
                            continue
                            
                        identitys.append(identity)
                        similaritys.append(sim)
                        valid_boxes.append(box)
                    
                    current_frame_identities = set(identitys)
        
        # --- 更新連續出現次數 ---
        # 1. 將當前畫面有偵測到的代號，連續次數 + 1
        for identity in current_frame_identities:
            consecutive_counts[identity] = consecutive_counts.get(identity, 0) + 1
            
        # 2. 將當前畫面「沒有」偵測到的代號，連續次數歸零 (直接從字典刪除)
        missing_identities = [id for id in consecutive_counts.keys() if id not in current_frame_identities]
        for id in missing_identities:
            del consecutive_counts[id]
        # ------------------------------

        # --- 檢查是否有離開事件 ---
        for identity in list(entry_times.keys()):
            if identity not in current_frame_identities and identity in last_seen_time:
                if (current_video_time - last_seen_time[identity]) > disappear_threshold:
                    departure_time = last_seen_time[identity]
                    entry_time = entry_times.pop(identity)
                    stay_duration = departure_time - entry_time
                    print(f"[{format_mm_ss(departure_time)}] 人員 {identity} 離開! (entry_time={format_mm_ss(entry_time)}, departure_time={format_mm_ss(departure_time)}, stay_duration={format_seconds(stay_duration)})")
                    # 將結果存成檔案
                    with open(log_filename, "a") as f:
                        f.write(f"{identity},{format_mm_ss(entry_time)},{format_mm_ss(departure_time)},{format_seconds(stay_duration)}\n")
        
        # 5. 追蹤出現時間與畫框
        for identity, sim, box in zip(identitys, similaritys, valid_boxes):
            
            # 根據是否為 unknown 決定對應的連續幀數門檻
            threshold = min_consecutive_unknown if identity == 'unknown' else min_consecutive_known
            
            # 檢查是否達到連續出現幀數門檻
            if consecutive_counts.get(identity, 0) >= threshold:
                # 若首次出現，或距離上次「真正出現」超過設定的秒數
                if identity not in last_seen_time or (current_video_time - last_seen_time[identity]) > disappear_threshold:
                    # 若尚未記錄 entry_time，則視為進入事件
                    if identity not in entry_times:
                        entry_times[identity] = current_video_time
                        print(f"[{format_mm_ss(current_video_time)}] 人員 {identity} 進入! entry_time={format_mm_ss(current_video_time)} (已連續 {consecutive_counts[identity]} 幀, 信心度: {sim:.2f})")
                    
                # 更新最後看見的時間 (只有達標時才更新，代表真正存在)
                last_seen_time[identity] = current_video_time
            
            # 畫框與標示文字
            color = (0, 255, 0) if identity != 'unknown' else (0, 0, 255)
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{identity} ({sim:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        # 6. 顯示即時畫面
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Face Recognition Tracking', display_frame)
        
        # 按下 'q' 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
def recognize_frame(face_manager, frame, similarity_threshold, min_unknown_face_area):
    identitys = []
    similaritys = []
    valid_boxes = []

    preds_list = face_manager.inference([frame])
    if len(preds_list) == 0:
        return identitys, similaritys, valid_boxes

    preds = preds_list[0]
    boxes = preds.boxes.data.cpu().numpy() if preds.boxes is not None else []
    keypoints = preds.keypoints.data.cpu().numpy() if preds.keypoints is not None else []
    if len(boxes) == 0:
        return identitys, similaritys, valid_boxes

    crops = []
    temp_valid_boxes = []
    for i in range(len(boxes)):
        box = boxes[i]
        lmk = keypoints[i][..., :-1]
        crops.append(face_manager.align(frame, lmk))
        temp_valid_boxes.append(box)

    if len(crops) == 0:
        return identitys, similaritys, valid_boxes

    embeds = face_manager.get_embedding_from_imgs(crops)
    raw_identitys, raw_similaritys = face_manager.recognize(
        embeds,
        similarityThresHold=similarity_threshold
    )

    for identity, sim, box in zip(raw_identitys, raw_similaritys, temp_valid_boxes):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if identity == 'unknown' and area < min_unknown_face_area:
            continue
        identitys.append(identity)
        similaritys.append(sim)
        valid_boxes.append(box)

    return identitys, similaritys, valid_boxes


def draw_recognition_result(frame, identitys, similaritys, valid_boxes):
    for identity, sim, box in zip(identitys, similaritys, valid_boxes):
        color = (0, 255, 0) if identity != 'unknown' else (0, 0, 255)
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{identity} ({sim:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def update_consecutive_counts(consecutive_counts, current_frame_identities):
    for identity in current_frame_identities:
        consecutive_counts[identity] = consecutive_counts.get(identity, 0) + 1

    missing_identities = [identity for identity in consecutive_counts.keys()
                          if identity not in current_frame_identities]
    for identity in missing_identities:
        del consecutive_counts[identity]


def main_split_videos():
    video_in_path = 'item2video_in.mp4'
    video_out_path = 'item2video_out.mp4'
    db_path = 'test_0_1'
    start_time_sec = 0.0

    similarity_threshold = 0.35
    min_unknown_face_area = 5000
    min_consecutive_known = 5
    min_consecutive_unknown = 32
    disappear_threshold = 2.0

    print("正在初始化人臉辨識系統與資料庫...")
    face_manager = CustomFaceManage(
        deviceDetect="cuda:0",
        deviceRecog="cuda:0",
        referencePath=db_path,
        voting=False
    )
    print("初始化完成，開始處理影片...")

    log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    with open(log_filename, "w") as f:
        f.write("id,entry_time,departure_time,stay_duration\n")

    entry_times = {}
    roi_pts = np.array([[1200, 350], [1250, 350], [1250, 400], [1200, 400]], np.int32)

    def process_video(video_path, mode):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return False

        if start_time_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
            print(f"{mode}: 已跳轉至影片 {start_time_sec} 秒處開始播放。")

        consecutive_counts = {}
        window_name = f"Face Recognition Tracking - {mode}"
        print(f"開始處理 {mode} 影片: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{mode} 影片播放完畢。")
                break

            cv2.fillPoly(frame, [roi_pts], (255, 255, 255))
            current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            identitys, similaritys, valid_boxes = recognize_frame(
                face_manager,
                frame,
                similarity_threshold,
                min_unknown_face_area
            )
            current_frame_identities = set(identitys)
            update_consecutive_counts(consecutive_counts, current_frame_identities)

            for identity, sim, _ in zip(identitys, similaritys, valid_boxes):
                threshold = min_consecutive_unknown if identity == 'unknown' else min_consecutive_known
                if consecutive_counts.get(identity, 0) < threshold:
                    continue

                if mode == "entry":
                    if consecutive_counts.get(identity, 0) == threshold:
                        entry_times.setdefault(identity, []).append(current_video_time)
                        print(f"[entry {format_mm_ss(current_video_time)}] 人員 {identity} 進入! "
                              f"(已連續 {consecutive_counts[identity]} 幀, 信心度: {sim:.2f})")
                else:
                    if consecutive_counts.get(identity, 0) == threshold and entry_times.get(identity):
                        entry_time = entry_times[identity].pop(0)
                        if not entry_times[identity]:
                            del entry_times[identity]
                        departure_time = current_video_time
                        stay_duration = departure_time - entry_time
                        print(f"[departure {format_mm_ss(departure_time)}] 人員 {identity} 離開! "
                              f"(entry_time={format_mm_ss(entry_time)}, departure_time={format_mm_ss(departure_time)}, "
                              f"stay_duration={format_seconds(stay_duration)})")
                        with open(log_filename, "a") as f:
                            f.write(f"{identity},{format_mm_ss(entry_time)},{format_mm_ss(departure_time)},{format_seconds(stay_duration)}\n")

            draw_recognition_result(frame, identitys, similaritys, valid_boxes)
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyWindow(window_name)
                return False

        cap.release()
        cv2.destroyWindow(window_name)
        return True

    if not process_video(video_in_path, "entry"):
        cv2.destroyAllWindows()
        return

    process_video(video_out_path, "departure")
    cv2.destroyAllWindows()


def main_watch_both_videos():
    video_in_path = 'item2video_in.mp4'
    video_out_path = 'item2video_out.mp4'
    db_path = 'test_0_1'
    start_time_sec = 0.0

    similarity_threshold = 0.35
    min_unknown_face_area = 5000
    min_consecutive_known = 5
    min_consecutive_unknown = 32
    disappear_threshold = 3.0

    print("正在初始化人臉辨識系統與資料庫...")
    face_manager = CustomFaceManage(
        deviceDetect="cuda:0",
        deviceRecog="cuda:0",
        referencePath=db_path,
        voting=False
    )
    print("初始化完成，開始同時處理進入與出去影片...")

    log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    with open(log_filename, "w") as f:
        f.write("id,entry_time,departure_time,stay_duration\n")

    cap_in = cv2.VideoCapture(video_in_path)
    cap_out = cv2.VideoCapture(video_out_path)

    if not cap_in.isOpened():
        print(f"無法開啟進入影片: {video_in_path}")
        cap_out.release()
        return
    if not cap_out.isOpened():
        print(f"無法開啟出去影片: {video_out_path}")
        cap_in.release()
        return

    if start_time_sec > 0:
        cap_in.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
        cap_out.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
        print(f"兩支影片皆跳轉至 {start_time_sec} 秒處開始播放。")

    entry_times = {}
    consecutive_counts_in = {}
    consecutive_counts_out = {}
    departure_last_seen = {}
    roi_pts = np.array([[1200, 350], [1250, 350], [1250, 400], [1200, 400]], np.int32)

    def process_frame(frame, current_video_time, mode, consecutive_counts):
        cv2.fillPoly(frame, [roi_pts], (255, 255, 255))

        identitys, similaritys, valid_boxes = recognize_frame(
            face_manager,
            frame,
            similarity_threshold,
            min_unknown_face_area
        )
        current_frame_identities = set(identitys)
        update_consecutive_counts(consecutive_counts, current_frame_identities)

        confirmed_identities = set()

        for identity, sim, _ in zip(identitys, similaritys, valid_boxes):
            threshold = min_consecutive_unknown if identity == 'unknown' else min_consecutive_known
            if consecutive_counts.get(identity, 0) < threshold:
                continue
            confirmed_identities.add(identity)

            if mode == "entry":
                if consecutive_counts.get(identity, 0) == threshold and identity not in entry_times:
                    entry_times[identity] = current_video_time
                    print(f"[entry {format_mm_ss(current_video_time)}] 人員 {identity} 進入! "
                          f"(已連續 {consecutive_counts[identity]} 幀, 信心度: {sim:.2f})")
            else:
                if identity in entry_times:
                    departure_last_seen[identity] = current_video_time
                if False and consecutive_counts.get(identity, 0) == threshold and identity in entry_times:
                    entry_time = entry_times.pop(identity)
                    departure_time = current_video_time
                    stay_duration = departure_time - entry_time
                    print(f"[departure {format_mm_ss(departure_time)}] 人員 {identity} 離開! "
                          f"(entry_time={format_mm_ss(entry_time)}, departure_time={format_mm_ss(departure_time)}, "
                          f"stay_duration={format_seconds(stay_duration)})")
                    with open(log_filename, "a") as f:
                        f.write(f"{identity},{format_mm_ss(entry_time)},{format_mm_ss(departure_time)},{format_seconds(stay_duration)}\n")

        if mode == "departure":
            missing_departure_identities = [
                identity for identity in departure_last_seen.keys()
                if identity not in confirmed_identities
            ]
            for identity in missing_departure_identities:
                last_seen_time = departure_last_seen[identity]
                if current_video_time - last_seen_time < disappear_threshold:
                    continue

                del departure_last_seen[identity]
                if identity in entry_times:
                    entry_time = entry_times.pop(identity)
                    departure_time = last_seen_time
                    stay_duration = departure_time - entry_time
                    print(f"[departure {format_mm_ss(departure_time)}] 人員 {identity} 離開! "
                          f"(entry_time={format_mm_ss(entry_time)}, departure_time={format_mm_ss(departure_time)}, "
                          f"stay_duration={format_seconds(stay_duration)})")
                    with open(log_filename, "a") as f:
                        f.write(f"{identity},{format_mm_ss(entry_time)},{format_mm_ss(departure_time)},{format_seconds(stay_duration)}\n")

        draw_recognition_result(frame, identitys, similaritys, valid_boxes)
        return frame

    in_finished = False
    out_finished = False
    print(f"開始處理進入影片: {video_in_path}")
    print(f"開始處理出去影片: {video_out_path}")

    while not in_finished or not out_finished:
        if not in_finished:
            ret_in, frame_in = cap_in.read()
            if ret_in:
                current_time_in = cap_in.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_in = process_frame(frame_in, current_time_in, "entry", consecutive_counts_in)
                cv2.imshow("Entry - video_in_path", cv2.resize(frame_in, (1280, 720)))
            else:
                in_finished = True
                print("進入影片播放完畢。")

        if not out_finished:
            ret_out, frame_out = cap_out.read()
            if ret_out:
                current_time_out = cap_out.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_out = process_frame(frame_out, current_time_out, "departure", consecutive_counts_out)
                cv2.imshow("Departure - video_out_path", cv2.resize(frame_out, (1280, 720)))
            else:
                out_finished = True
                print("出去影片播放完畢。")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_watch_both_videos()
