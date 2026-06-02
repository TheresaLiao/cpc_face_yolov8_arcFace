CodeBase = "."

import torch
import numpy as np
import cv2
import torchvision
from skimage import transform as trans
from iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from tqdm import tqdm
import argparse
import time
import json
from ultralytics import YOLO
import glob,os
from modules import VideoCapture as cap
import datetime
torch.cuda.empty_cache()
from collections import defaultdict
# import matplotlib.pyplot as plt

import requests
from PIL import Image, ImageDraw, ImageFont
from requests.auth import HTTPBasicAuth
import subprocess
import pymssql


#write_db = False
match_time = 2  # 要連續偵測到5次才放行


#urls = ['rtsp://root:vivoadm1234@10.1.1.221:554/live.sdp']     #office
#urls = ['rtsp://admin:123456@10.1.1.85:554/profile1']   #office
#urls = ['rtsp://admin:d50215021@192.168.11.4:8554/Media/Live/Normal?camera=C_1&streamindex=1']     #nvr
#urls = ['rtsp://Admin:123456@192.168.11.162:7070']      #ipcam
urls = ['test.mp4']     #test


# UI
APIServerIP = "127.0.0.1"  # CPC_Tainan host-ip: 192.168.11.200
PORT = "8000"
DBserver = '127.0.0.1:1533'
#APIServerIP = "10.1.1.10"  # CPC_Tainan host-ip: 192.168.11.200
#PORT = "8000"
#DBserver = '10.1.1.10:1533'


def save2History(tableName, personnel_id, personnel_name, company_name, project_start_date, project_end_date, screenshot_path):
    # conn = pymssql.connect(server=server, database=database, user=username, password=password)
    conn = pymssql.connect(DBserver, 'cpc1_db', 'cpc1@Tainan01', 'cpcTNNdb')
    cursor = conn.cursor()
    storage_time = datetime.datetime.now()
    insert_query = f"INSERT INTO {tableName} (storage_time, personnel_id, personnel_name, company_name, project_start_date, project_end_date, screenshot_path) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    data = (storage_time, personnel_id[:-1], personnel_name, company_name, project_start_date, project_end_date, screenshot_path)
    #print("----",data)
    cursor.execute(insert_query, data)
    conn.commit()
    conn.close()


def updateworkingtime():
    print(datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S"))

    conn = pymssql.connect(DBserver, 'cpc1_db', 'cpc1@Tainan01', 'cpcTNNdb')  
    cursor = conn.cursor()
    if not cursor:
        print('数据库连接失败！')

    cursor.execute(f"SELECT day FROM doormanage_weekdays WHERE is_workday=1")
    result = cursor.fetchall()
    #print("------------------------------------ result =", result)
    workdays = [result[i][0] for i in range(len(result))]
    print("------------------------------------ workdays =", workdays)
    '''
    now_weekday = datetime.datetime.now().weekday()+1
    print(now_weekday)
    print(str(now_weekday) in workdays)
    '''
    
    cursor.execute(f"SELECT start_time,stop_time FROM doormanage_runningtime")
    result = cursor.fetchall()
    #print("------------------------------------ result =", result)
    start_time = result[0][0]
    stop_time = result[0][1]
    print("------------------------------------ start_time =", start_time, "stop_time =", stop_time)
    #global faceDetect_start     #debug
    #print("--- faceDetect_start =",faceDetect_start,"---")    #debug
    #global last_faceDetect_start     #debug
    #print("--- now-last_faceDetect_start =",time.time()-last_faceDetect_start,"---")    #debug
    '''
    now_time = datetime.datetime.now().time()
    print(now_time)
    print(now_time > start_time)
    print(now_time < stop_time)
    '''

    return workdays, start_time, stop_time
    


#------------------------------------------------
def date2EvaluateNum(date_string):
    '''
    date_string => ex: "20230220"

    # **********************************
    # [Usage]
    start = date2EvaluateNum("20230201")
    end = date2EvaluateNum("20230210")
    current = time.time()
    if start is not None and end is not None:
        print(current > start and current < end)

    # **********************************

    '''
    if len(date_string) != 8:
        print(f"{date_string} - 日期格式錯誤")
        return None
    else:
        year = date_string[:4]
        month = date_string[4:6]
        date = date_string[6:8]
        date_str = f'{month}-{date}-{year}'
        date_object = datetime.datetime.strptime(date_str, '%m-%d-%Y').date()
        eva_num = time.mktime(date_object.timetuple())
    return eva_num

def process_imgpath2info(img_dir,img_path):
    '''
    img_path 結構 => ID_名子_公司名稱_起日_終日_流水號.png
    img_dir => aaa/bbb/
    '''
    data_types = [".jpg",".png","json"]
    if img_path[-4:] not in data_types:
        print(f"{img_path} - 不是影像資料")
        return None
    img_path_type = img_path[-4:]
    img_path = img_path[:-4]
    imgpath_arr = img_path.split("_")
    if len(imgpath_arr) != 6:
        print(f"{img_path} - 影像資料錯誤，不予加入辨識資料集......")
        return None

    else:
        ID,name,company,start_date,end_date,water_number = imgpath_arr
        if img_dir is not None:
            img_path = img_dir + img_path + img_path_type
            feature_path = img_path[:-4] + ".json"
        info = [ID,name,company,start_date,end_date,water_number,img_path,feature_path]
        return info

# def filtering_date(currentNum,imglist):
    # '''
    # 目的：[以現在日期過濾掉已經過期或還沒生效的list]
    # currentNum => ex: 1676822400.0 ; type: str/int/float
    # imglist => ex: [XX.png, XX.png, ...] ; type: list
    # '''
    # currentNum = int(currentNum)
    # img_num = len(imglist)
    # imglistFiltering = []
    # for imgIndex in range(img_num):
    #     img_path = imglist[imgIndex]
    #     img_path_org = img_path
    #     if "/" in img_path:
    #         img_path = img_path.split("/")[-1]
    #     pathArr = img_path.split("_")
    #     img_path_start,img_path_end = pathArr[3],pathArr[4]
    #     startNum = date2EvaluateNum(img_path_start)
    #     endNum = date2EvaluateNum(img_path_end)
    #     if currentNum >= startNum and currentNum <= endNum:
    #         imglistFiltering.append(img_path_org)
    # return imglistFiltering

class SouceReading(object):
    def __init__(self, 
                 use_gst=False,
                 cycle=False,
                 sourcesList=None,
                 expectFPS = None,
                 ):
        self.use_gst = use_gst
        self.cycle = cycle
        self.sourcesList = sourcesList
        self.imgs = [np.zeros((2160,3840,3),dtype='uint8') for _ in self.sourcesList]
        self.rets = [False for _ in self.sourcesList]
        self.vidcaps = self.multi_video_load()
        
        if not expectFPS is None:
            self.expectFPS = expectFPS
            print(f"Expect FPS : {expectFPS}")
            self.init_fixed_fps()
        print("Source num : ", len(self.vidcaps))

    def init_fixed_fps(self):
        self.videoFPSs = [vidcap.get(cv2.CAP_PROP_FPS) for vidcap in self.vidcaps]
        self.skipFramessLooply=  [int((videoFPS/self.expectFPS)-1) 
                                 for videoFPS in self.videoFPSs]
        self.processing_time_each_frame = 1/(self.expectFPS)

    def manage_looply(self,t1_of_loop):
        counts = [0] * len(self.vidcaps)
        counts = [sum(1 for _ in range(skipFramesLooply) if vcap.read()[0] is not None) 
                            for vcapIdx, (vcap, skipFramesLooply) in 
                                enumerate(zip(self.vidcaps, self.skipFramessLooply))]
        
        print("skip counts : ",counts)
        t_diff = time.time() - t1_of_loop
        if t_diff < self.processing_time_each_frame:
            time.sleep(self.processing_time_each_frame - t_diff)
        print(f"FPS : {1/(time.time()-t1_of_loop)}")   


    def multi_video_load(self):
        vidcaps = []
        for video_index,source in enumerate(self.sourcesList):
            if self.sourcesList[video_index].startswith("rtsp"):
                if self.use_gst:
                    vidcaps.append(cap.VideoCapture(source,0,0,0))
                else:
                    vidcaps.append(cap.VideoCapture(source))
            else:
                vidcaps.append(cv2.VideoCapture(source))
        return vidcaps
    def sources_read(self):
        for video_index,source in enumerate(self.sourcesList):
            ret, image = self.vidcaps[video_index].read()
            self.rets[video_index] = ret
            if not ret and self.cycle:
                self.vidcaps[video_index].release()
                time.sleep(0.001)
                if self.sourcesList[video_index].startswith("rtsp"):
                    if self.use_gst:
                        self.vidcaps[video_index] = cap.VideoCapture(self.sourcesList[video_index],0,0,0)
                    else:
                        self.vidcaps[video_index] = cap.VideoCapture(self.sourcesList[video_index])
                else:
                    self.vidcaps[video_index] = cv2.VideoCapture(self.sourcesList[video_index])
                continue
            elif not ret and not self.cycle:
                self.vidcaps[video_index].release()
            else:
                self.imgs[video_index] = image

                # for testing
                #self.imgs[video_index] = cv2.imread("2024_05_14_14_01_555_similarity_0.88.png")
                #self.imgs[video_index] = cv2.imread("D123418013_555_明琨工程有限公司_20221009_20240531_1.png")
                #self.imgs[video_index] = cv2.imread("2024_05_15_08_55_張簡武男_similarity_0.91.png")


        return self.imgs,self.rets
    
def get_ip_from_rtspString(rtspString):
    rtspStringArr = rtspString.split("/")
    max_dots = 0 ;max_index = -1
    for index, item in enumerate(rtspStringArr):
        dots_count = item.count('.')
        if dots_count > max_dots:
            max_dots = dots_count
            max_index = index
    return rtspStringArr[max_index]

def list_unique(list_data):
    return list(set(list_data))

def putText_chinese(img,textC,pos):
    imgPil = Image.fromarray(img)                # 將 img 轉換成 PIL 影像
    draw = ImageDraw.Draw(imgPil)                # 準備開始畫畫
    draw.text(pos, textC, fill=(0, 0, 255), font=font)  # 畫入文字，\n 表示換行
    img = np.array(imgPil)                       # 將 PIL 影像轉換成 numpy 陣列
    return img



def show_imgs(imgs,new_size=None):
    for imgIdx,img in enumerate(imgs):
        if new_size:
            img = cv2.resize(img,new_size)
        cv2.imshow(f'{imgIdx}', img)
    if cv2.waitKey(1) == ord('q'):
        exit()

def get_model(name, **kwargs): ## arcface
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)

    else:
        raise ValueError()


class FaceManage(object):
    def __init__(self, 
                 deviceDetect = "cuda:0",
                 deviceRecog  = "cuda:0",
                 detect_Model_file_path=f'{CodeBase}/weights/yolov8n-face.pt',
                 embedModelPath = f"{CodeBase}/weights/glint360k_cosface_r18_fp16_0.1/backbone.pth",
                 referencePath = None,
                 referencePathAutoAdd = None,
                 voting = False
                 ):

        self.deviceDetect = deviceDetect
        self.deviceRecog = deviceRecog
        self.model_det = YOLO(detect_Model_file_path)  # load an official model
        self.tform = trans.SimilarityTransform()
        self.src = np.array([[30.2946, 51.6963],
                             [65.5318, 51.5014],
                             [48.0252, 71.7366],
                             [33.5493, 92.3655],
                             [62.7299, 92.2041]], dtype=np.float32)
        self.src[:, 0] += 8.0

        self.embedModelPath = embedModelPath
        self.embedModel = self.load_embed_model()
        self.embedModel.to(self.deviceRecog)
        self.embedModel.eval()
        self.featureDim = 512
        self.referencePath = referencePath
        self.referencePathAutoAdd = referencePathAutoAdd
        self.prepare_reference_embeddings(self.referencePath)
        # self.prepare_reference_embeddings_aligned(self.referencePath)
        self.embeds_ref,self.ref_persons,self.fnamesRef = self.get_reference_embeds_from_folder_jsons(self.referencePath)
        self.saveHistory = defaultdict(lambda:0)

        if voting:
            self.match_list = []
            self.simularity_list = []
            self.img_most_simular_list = []

    def vote(self,identify,similarity,img_most_simular,match_time=match_time):
        if not len(identify): # init
            self.match_list = []
            self.simularity_list = []
            self.img_most_simular_list = []
            return None,None,None


        self.match_list.append(identify[0])
        self.simularity_list.append(similarity[0])
        self.img_most_simular_list.append(img_most_simular[0])
        if len(self.match_list) >= match_time:
            self.match_list = self.match_list[-match_time:]
            self.simularity_list = self.simularity_list[-match_time:]
            self.img_most_simular_list = self.img_most_simular_list[-match_time:]


            match_list_uni = list_unique(self.match_list)
            if len(match_list_uni) != 1:  # 連續n次結果相同才發報
                return None,None,None
            else:
                if self.match_list[0] == "unknown":
                    img_most_simular_report = "None"
                    simularity_report = 0
                    identity_report = "unknown"
                else:
                    img_most_simular_report = self.img_most_simular_list[np.argmax(self.simularity_list)]
                    simularity_report = np.mean(self.simularity_list)
                    identity_report = self.match_list[0]
                
                self.match_list = []
                self.simularity_list = []
                self.img_most_simular_list = []

                return identity_report,simularity_report,img_most_simular_report
        else:
            return None,None,None


    # def prepare_reference_embeddings_aligned(self,imgDBpath):
    #     imgPaths_NeedInference,imgs = self.findNeedEmbeddingImgsAligned(imgDBpath)
    #     for img,imgPath in zip(imgs,imgPaths_NeedInference):
    #         cv2.imwrite('tmp.png',img)
    #         exit()
    #         embed = self.get_embedding_from_imgs([img])[0] if len(crops) else []
    #         self.saveJson([imgPath],[embed],verbose =True) if len(crops) else None
    #     return 0

    def prepare_reference_embeddings(self,imgDBpath):
        if imgDBpath is None:
            print("prepare_reference_embeddings: reference path is None")
            return 0
        if not os.path.isdir(imgDBpath):
            print(f"prepare_reference_embeddings: reference path not found: {imgDBpath}")
            return 0

        imgPaths_NeedInference,imgs = self.findNeedEmbeddingImgs(imgDBpath)
        if not imgPaths_NeedInference:
            print(f"No new images found for embedding in {imgDBpath}")
            return 0

        for img,imgPath in zip(imgs,imgPaths_NeedInference):
            if img is None:
                print(f"Cannot load image: {imgPath}")
                continue
            img = resize_and_fill(img, (1920, 1080))  
            preds = self.inference([img])
            print(f"\n--- Debug: {imgPath.split('/')[-1]} ---")
            print(f"preds type: {type(preds)}")
            print(f"preds: {preds}")
            print(f"-------------------")
            inference_informations = self.postprocess(preds)
            crops,_ = self.get_aligned_crops_max([img],inference_informations,NoneAsBlack=False)

            if not crops:
                print(f"File cannot find face : {imgPath.split('/')[-1]}")
                continue

            aligned_crop = crops[0]
            cv2.imwrite(f"{CodeBase}/aligned/{imgPath.split('/')[-1][:-4]}full.png",img)
            cv2.imwrite(f"{CodeBase}/aligned/{imgPath.split('/')[-1][:-4]}_face_aligned.png",aligned_crop)
            embed_arr = self.get_embedding_from_imgs([aligned_crop])
            if len(embed_arr) == 0:
                print(f"Embed generation failed: {imgPath.split('/')[-1]}")
                continue
            embed = embed_arr[0]
            self.saveJson([imgPath],[embed],verbose=True)
            print(f"Saved json for {imgPath}")
        return 0

    def show_crops(self,crops):
        for imgIdx,crop in enumerate(crops):
            if crop is None:
                continue
            cv2.imshow(f'crop_{imgIdx}', crop)
        if cv2.waitKey(1) == ord('q'):
            exit()

    def inference(self,imgs):
        return self.model_det(imgs,verbose=False)

    def postprocess(self,preds,getMaxFace = True,maxFaceArea = 500):
        preds = preds[:-1] ## less one is tensor, here is unnecessary

        inference_informations = []
        for predIdx,pred in enumerate(preds):
            keypoints = pred.keypoints.data.cpu().numpy()
            boxes = pred.boxes.data.cpu().numpy()
            if not len(boxes):
                inference_informations.append([None,None])
                continue
            if getMaxFace:
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]*boxes[:,1])
                maxFaceIdx = np.argmax(areas)
                maxArea = np.max(areas)
                if maxArea <= maxFaceArea:
                    print(f"max face area only {maxArea} (Thres:{maxFaceArea})  so skip")
                    inference_informations.append([None,None])
                    continue
                inference_informations.append([boxes[maxFaceIdx],
                                              keypoints[maxFaceIdx]])
        return inference_informations

    def align(self,img,lmk):
        self.tform.estimate(lmk, self.src)
        M = self.tform.params[0:2, :].astype(np.float32)
        img_align = cv2.warpAffine(img,
                                M, (112, 112),
                                borderValue=0.0)     
        # img_align = cv2.cvtColor(img_align, cv2.COLOR_BGR2RGB)
        return img_align

    def get_aligned_crops_max(self,imgs,inference_informations,NoneAsBlack=True):
        crops = [];belongline=[]
        for lineIdx,(img,inference_information) in enumerate(zip(imgs,inference_informations)):
            if inference_information[0] is None:
                if NoneAsBlack:
                    crops.append(np.zeros((112,112,3),dtype="uint8"))
                    belongline.append(lineIdx)
            else:
                # print(inference_information[1][...,:-1])
                crops.append(self.align(img,inference_information[1][...,:-1]))
                belongline.append(lineIdx)
        return crops,belongline

    def drawDetectBBOXes(self,imgs,inference_informations,
                              plot_landmarks = True,
                              plot_confidence = False,
                              ):
        img_plots = []
        for detIdx in range(len(imgs)):
            inference_information = inference_informations[detIdx]
            img_plot = imgs[detIdx].copy()
            if inference_information[0] is None:
                img_plots.append(img_plot)   
                continue

            box = inference_information[0]
            lmks = inference_information[1]
            # print(detIdx,img_plot.shape)
            # print(box)
            cv2.rectangle(img_plot, (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])),
                                    (0, 255, 0), 2)
            if plot_landmarks:
                for lmk in lmks:
                    cv2.circle(img_plot,(int(lmk[0]), int(lmk[1])),
                                            5, (255, 255, 255), -1)
            if plot_confidence:
                cv2.putText(img_plot, f"{round(float(box[4]),2)}", 
                                      (int(box[0]), int(box[1]-30)), cv2.FONT_HERSHEY_SIMPLEX,
                                      2, (0, 0, 0), 2, cv2.LINE_AA) 
            img_plots.append(img_plot)          
        return img_plots

    # def findNeedEmbeddingImgsAligned(self,reference_imgs_path):
    #     imgPaths = []
    #     imgForms = ["png","jpg"]
    #     shouldContain = "face_aligned"
    #     for imgForm in imgForms:
    #         imgPaths_tmp = glob.glob(f"{reference_imgs_path}/*.{imgForm}")
    #         for imgPath in imgPaths_tmp:
    #             if not shouldContain in imgPath:
    #                 continue
    #             imgPaths.append(imgPath)
                
    #     imgPaths_NeedInference = []
    #     imgs = []
    #     for imgPath in imgPaths:
    #         jsonPath = imgPath[:-3]+"json"
    #         if not os.path.exists(jsonPath):
    #             img = cv2.imread(imgPath)
    #             imgs.append(img)
    #             imgPaths_NeedInference.append(imgPath)
    #     return imgPaths_NeedInference,imgs
                

    def findNeedEmbeddingImgs(self,reference_imgs_path):
        imgPaths = []
        imgForms = ["png","jpg","jpeg","PNG","JPG","JPEG"]
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
            jsonPath = imgPath[:-3]+"json"
            if not os.path.exists(jsonPath):
                img = cv2.imread(imgPath)
                if img is None:
                    print(f"Cannot load image file: {imgPath}")
                    continue
                imgs.append(img)
                imgPaths_NeedInference.append(imgPath)
        return imgPaths_NeedInference,imgs
                
    def saveJson(self,imgPaths,embeds,verbose=False):
        for imgPath,embed in zip(imgPaths,embeds):
            if embed is None:
                continue
            data = {
                "filename" : imgPath,
                "embed" : embed.tolist()               
            }
            jsonPath = imgPath[:-3]+"json"
            json_dir = os.path.dirname(jsonPath)
            if json_dir and not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
            if verbose:
                print(f"[Write json] : {jsonPath}")
            with open(jsonPath, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file,ensure_ascii=False)
            # print("Save embedding Json :" , jsonPath)


    def AutoTimelyUpdateReferenceSet(self,count=None,countly=500,filtering_by_date=False):
        
        if count % countly == 0:
            if self.referencePath is None:
                return 0
            elif os.path.isdir(self.referencePath):
                print("loading embedding mode : from jsons")
                self.reference_folder = self.referencePath
                self.prepare_reference_embeddings(self.reference_folder)
                self.embeds_ref,self.identitys_ref,self.fnamesRef = self.get_reference_embeds_from_folder_jsons(self.reference_folder,
                                                                                                                filtering_by_date=filtering_by_date)
                print(f"update of count : {count}")


    def saveRecognizeResult2DB(self,crops=None,aligned_crops=None,
                               identitys=None,similaritys=None,urls=None,
                               belongline=None,
                               minSimilarity=0.4,maxSimilarity=0.6,verbose=True,
                               save2Json=False, embeds = None,saveRangeSamePerson=None,
                               ):
        
        if self.referencePath is None:
            print("you didnt give the referencePath, how can I fucking update ?")
            return 0
        if self.referencePath.endswith(".h5"):
            print("h5 mode is not finished ..... qq")
            return 0

        dataNum = len(crops)
        for dataIdx in range(dataNum):
            similarity = similaritys[dataIdx]
            if similarity > maxSimilarity:
                continue
            if similarity < minSimilarity:
                continue
            crop = crops[dataIdx]
            if not 0 in crop.shape:
                aligned_crop = aligned_crops[dataIdx]
                identity = identitys[dataIdx]
                belonglineIdx = belongline[dataIdx]

                if not saveRangeSamePerson is None:
                    if time.time() - self.saveHistory[identity] > saveRangeSamePerson:
                        self.saveHistory[identity] = time.time()
                    else:
                        print(f"[time in range so skip] : {identity}")
                        continue
                    
                url = urls[belonglineIdx]
                savePath = None
                if url.startswith("rtsp://"):
                    url_string = url.replace("/","_").replace(":","_").replace("@","_")
                    now = str(datetime.datetime.now()).replace(" ","-").replace(":","-")
                    savePath = f"{self.referencePathAutoAdd}/{identity}/{url_string}_{now}_autoAdd"
                folder = os.path.dirname(f"{savePath}_face_aligned_retinaface_1080p.png")
                if not os.path.exists(folder):
                    os.mkdir(folder)
                cv2.imwrite(f"{savePath}_face_aligned_retinaface_1080p.png",aligned_crop) # [:,:,::-1]

                print(f"{savePath}_face_aligned_retinaface_1080p.png")
                if save2Json:
                    embed = embeds[dataIdx]
                    self.saveJson([f"{savePath}_face_aligned_retinaface_1080p.png"],[embed])
                if verbose:
                    print(f"auto save image : {savePath} to DB !!!!!!!!!!!!!!!!!")
        return 0


    def get_reference_embeds_from_folder_jsons(self,folder,
                                               get_name_from_filename=True,filtering_by_date=False):

        jsonFiles = glob.glob(folder+"/*.json")
        corresponding_persons = []
        reference_embeds = []

        if not len(jsonFiles):
            print("No any reference embedding json so exit()")
            exit()
        for jsonFile in jsonFiles:
            info = json.load(open(jsonFile))
            name = info['filename']

            if filtering_by_date:
                pathArr = name.split("/")[-1].split("_")
                img_path_start,img_path_end = pathArr[3],pathArr[4]
                startNum = date2EvaluateNum(img_path_start)
                endNum = date2EvaluateNum(img_path_end)
                if currentNum > startNum or currentNum > endNum:
                    print(f"[Out of date] : {name}")
                    continue

            if get_name_from_filename:
                name = name.split("/")[-1].split("_")[1]
            corresponding_persons.append(name)
            reference_embed = torch.from_numpy(np.array(info['embed']))
            reference_embeds.append(reference_embed.unsqueeze(0))
        reference_embeds = torch.cat(reference_embeds).float()
        return reference_embeds,corresponding_persons,jsonFiles

    def get_embedding_from_h5(self):
        hf =  h5py.File(self.reference_h5_path, 'r')
        hf_keys = list(hf.keys())
        embeddings = []
        identities = []
        print("loading reference embedding .....")
        pbar = tqdm(total=len(hf_keys))
        for keyIdx,hf_key in enumerate(hf_keys):
            pbar.update(1)
            embedding = torch.tensor(hf[hf_key]['embedding'])
            identity = np.array(hf[hf_key]['identity']).tolist().decode(encoding="utf-8")
            embeddings.append(embedding.unsqueeze(0))
            identities.append(identity)
            # if keyIdx > 100:
            #     break
        pbar.close()
        return torch.cat(embeddings),identities

    def get_embedding_from_imgs(self,aligned_crops,
                                BGR2RGB = True,normalize = True,
                                show_calculate_time = False,
                               ):
        if not len(aligned_crops):
            return []

        imgs_scaled = torch.empty((len(aligned_crops),3,112,112))
        for imgIdx,img in enumerate(aligned_crops):
            # img = cv2.resize(img, (112, 112))

            if BGR2RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if normalize:
                img = img.astype(float)
                img /= 255
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            imgs_scaled[imgIdx]=img
        imgs_scaled = imgs_scaled.to(self.deviceRecog)
        if show_calculate_time:
            t1 = time.time()
        
        embeds = None
        with torch.no_grad():
            embeds = self.embedModel(imgs_scaled).cpu().numpy()  
        if show_calculate_time:
            t2 = time.time()
            print("embed take time : " , t2-t1)
        return embeds

    def load_embed_model(self):
        embedModel = get_model("r18", fp16=False)
        embedModel.load_state_dict(torch.load(self.embedModelPath))
        return embedModel

    def recognize(self,embeds,similarityThresHold=0.4,returnMostSimiar=False):
        if not len(embeds):
            if returnMostSimiar:
                return [],[],[]
            return [],[]
        distmats = self.compare_encodings(embeds,self.embeds_ref)
        identitys,similaritys = np.argmax(distmats,axis=-1),np.max(distmats,axis=-1)
        identitys = identitys.tolist()
        if returnMostSimiar:
            imgMostSimlar = [self.fnamesRef[identity] for identity in identitys] if len(identitys) else []
        
        similaritys /= self.featureDim
        for dataIdx in range(len(similaritys)):
            if similaritys[dataIdx] <= similarityThresHold:
                similaritys[dataIdx] = 0
                identitys[dataIdx] = "unknown"
            else:
                identitys[dataIdx] = self.ref_persons[identitys[dataIdx]]

        if returnMostSimiar:
            return identitys,similaritys,imgMostSimlar
        return identitys,similaritys

    def drawRecognizeResult(self,imgs,inference_informations,
                            identitys,similaritys,belongline,
                            plot_confidence=False,
                            plot_landmarks=False,
                            plot_similarity=True,
                            fontSize = 0.8
                            ):


        for identity,similarity,inference_information,belonglineIdx in zip(identitys,similaritys,inference_informations,belongline):
            bbox = inference_information[0]
            lmks = inference_information[1]
            if bbox is None:
                continue

            color = (0,0,0)
            if identity != "unknown":
                color = (255,0,0)

            cv2.rectangle(imgs[belonglineIdx], (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])),
                                    color, 2)
            if plot_similarity:
                cv2.putText(imgs[belonglineIdx], f"similarity: {int(similarity*100)}", 
                            (int(bbox[0]), int(bbox[3]+60)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontSize, color, 2, cv2.LINE_AA)                 
            if plot_landmarks:
                lmks = dets[detIdx]["landms"][dataIdx]
                for lmk in lmks:
                    cv2.circle(imgs[belonglineIdx],(int(lmk[0]), int(lmk[1])), 30, (0, 255, 255), 3)
            if plot_confidence:
                cv2.putText(imgs[belonglineIdx], f"{round(float(bbox[4]),2)}", 
                            (int(bbox[0]), int(bbox[3]+90)), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, 2, cv2.LINE_AA) 
        return imgs
    
    def compare_encodings(self, embeds,embeds_ref):
        return np.dot(embeds, embeds_ref.T)


def resize_and_fill(image, target_size):
    height, width, _ = image.shape
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    background[:new_height,:new_width] = resized_image
    return background