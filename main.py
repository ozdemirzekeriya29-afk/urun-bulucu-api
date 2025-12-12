from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - DENGELİ MOD V7")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    # 1. RESMİ OKU
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlük (Detayları kaybetmemek için 1000 iyi bir standart)
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Griye Çevir + Kontrast Artır (CLAHE)
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    
    if des1 is None: return None, 0

    en_yuksek_skor = 0
    bulunan_kod = "Bulunamadı"

    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            db_path = os.path.join(KLASOR, dosya)
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE) # Gri okumak yeterli
            if db_img is None: continue

            # Veritabanı resmini boyutlandır
            h, w = db_img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                new_h = int(h * scale)
                db_img = cv2.resize(db_img, (1000, new_h))

            db_img = clahe.apply(db_img)
            kp2, des2 = sift.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                # Eşleştirme
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    # --- DÜZELTME 1: ORAN TESTİ ---
                    # 0.60 çok katıydı, hiçbir şeyi beğenmiyordu.
                    # 0.70 endüstri standardıdır. Işık farkını tolere eder.
                    if m.distance < 0.70 * n.distance:
                        iyi_eslesmeler.append(m)
                
                # --- DÜZELTME 2: MİNİMUM NOKTA SAYISI ---
                # 25 nokta bulmak zordur. 12 nokta "Geometrik olarak"
                # kusursuz hizalanıyorsa, o ürün o üründür.
                if len(iyi_eslesmeler) >= 12:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    # Geometri Doğrulama (RANSAC)
                    # Yanlış ürünleri eleyen ASIL KAHRAMAN budur.
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        # Geometriye uyan nokta sayısı
                        skor = sum(mask.ravel().tolist())
                        
                        # Eğer geometriye uyan nokta sayısı, toplam iyi noktalara oranla çok düşükse
                        # (Örn: 100 nokta buldum ama sadece 5'i hizalı) bu yanlıştır.
                        # Ama biz basit skor takibi yapalım:
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
    return {"durum": "aktif", "mod": "DENGELİ MOD (V7)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # --- FİNAL KARAR ---
    # 12 tane "Geometrik Olarak Kusursuz" nokta varsa kabul et.
    # Rastgele ürünlerde bu sayı genelde 4-5 çıkar.
    # Doğru ürünlerde 15-50 arası çıkar.
    LIMIT = 12
    
    if skor >= LIMIT:
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        mesaj = f"Yetersiz Veri ({skor})"
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
