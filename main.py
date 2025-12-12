from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma Servisi")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # RAM tasarrufu için boyutlandırma (Ama kaliteyi çok düşürmeden)
    # 800 yerine 1000 yaptık ki detaylar kaybolmasın
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Griye çevir + Kontrast Artır
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    
    # SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    
    if des1 is None: return None, 0

    en_yuksek_skor = 0
    bulunan_kod = "Bulunamadı"

    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            db_path = os.path.join(KLASOR, dosya)
            # Gri modda oku
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
            
            if db_img is None: continue
            
            # Veritabanı resmini de optimize et
            h, w = db_img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                new_h = int(h * scale)
                db_img = cv2.resize(db_img, (1000, new_h))

            db_img = clahe.apply(db_img)
            
            kp2, des2 = sift.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 2:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    # BURASI ÇOK ÖNEMLİ: 0.75 yerine 0.7 yaptık.
                    # Bu, "Az benzesin yeter" yerine "Çok benzesin" demektir.
                    if m.distance < 0.7 * n.distance:
                        iyi_eslesmeler.append(m)
                
                # RANSAC ile Geometrik Doğrulama
                if len(iyi_eslesmeler) >= 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        skor = sum(mask.ravel().tolist())
                        if skor > en_yuksek_skor:
                            en_yuksek_skor = skor
                            bulunan_kod = dosya.split(".")[0]

            del db_img, kp2, des2
        except:
            pass

    gc.collect()
    return bulunan_kod, en_yuksek_skor

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mesaj": "Sunucu Kalibre Edildi (V3)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # EŞİK DEĞERİ (Threshold)
    # Yanlış ürün bulmasın diye 6'dan 8'e çektik.
    ESIK = 8 
    
    if skor >= ESIK:
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        # Eğer skor varsa ama düşükse yine de kullanıcıya bilgi verelim (Test için)
        mesaj = "Benzerlik düşük." if skor > 0 else "Eşleşme yok."
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
