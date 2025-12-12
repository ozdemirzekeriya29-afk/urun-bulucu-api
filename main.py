from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - Final V4")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    # Resmi Oku
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlüğü makul seviyeye çek (1000px)
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Griye Çevir + CLAHE (Kontrast Artırma)
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    
    # SIFT Algoritması
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
            
            if des2 is not None and len(des2) > 10:
                # FLANN Eşleştirici
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    # Oran Testi: 0.7 (Ne çok sıkı ne çok gevşek)
                    if m.distance < 0.7 * n.distance:
                        iyi_eslesmeler.append(m)
                
                # --- İLK ELEME: Nokta Sayısı ---
                # Eğer en az 12 tane çok iyi nokta bulamadıysa, 
                # bu resim o resim değildir. Hiç geometriye girme.
                if len(iyi_eslesmeler) >= 12:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    # RANSAC Eşik Değeri: 4.0 (Noktaların ne kadar hizalı olması gerektiği)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    
                    if mask is not None:
                        # Sadece geometriye uyan noktaları say (Gerçek Skor)
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
    return {"durum": "aktif", "mesaj": "Sunucu V4 - Eşik 35"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # --- FİNAL BARAJI: 35 PUAN ---
    # Gerçek ürünler genelde 50-100 arası alır.
    # Rastgele ürünler 10-20 arası alır.
    ESIK = 35 
    
    if skor >= ESIK:
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        # Kullanıcıya skoru gösterelim ki neden bulamadığını anlasın
        if skor > 0:
            mesaj = f"Benzerlik Yetersiz (Skor: {skor}, Gereken: {ESIK})"
        else:
            mesaj = "Eşleşme Yok"
            
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
