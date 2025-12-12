from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma Servisi - Sıkı Mod")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    # Gelen resmi oku
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlüğü biraz daha net tutuyoruz (Detayları görebilsin diye)
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Gri Tonlama + Kontrast Eşitleme (CLAHE)
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    
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
            
            if des2 is not None and len(des2) > 5:
                # FLANN Parametreleri
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    # --- KRİTİK AYAR 1: SIKI EŞLEŞME ---
                    # 0.75 yerine 0.65 yaptık. Sadece "Çok benzeyen" noktaları alır.
                    if m.distance < 0.65 * n.distance:
                        iyi_eslesmeler.append(m)
                
                # --- KRİTİK AYAR 2: MİNİMUM NOKTA SAYISI ---
                # En az 8 nokta tutmazsa hiç geometri kontrolü yapma.
                if len(iyi_eslesmeler) >= 8:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    # RANSAC ile geometriyi doğrula
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        # Maske içindeki başarılı nokta sayısı
                        skor = sum(mask.ravel().tolist())
                        
                        if skor > en_yuksek_skor:
                            en_yuksek_skor = skor
                            bulunan_kod = dosya.split(".")[0]

            del db_img, kp2, des2 # Hafıza Temizliği
        except:
            pass

    gc.collect()
    return bulunan_kod, en_yuksek_skor

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mesaj": "Sunucu SIKI MODDA Calisiyor"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # --- KRİTİK AYAR 3: FİNAL EŞİK DEĞERİ ---
    # Bu değeri 15'e çıkardık.
    # Eğer skor 15'ten azsa "Bulunamadı" de.
    # Rastgele ürünler genelde 4-8 arası puan alır, elenirler.
    ESIK = 15
    
    if skor >= ESIK:
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        mesaj = f"Yetersiz Benzerlik ({skor})"
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
