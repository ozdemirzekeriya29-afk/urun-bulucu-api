from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os

# --- UYGULAMA AYARLARI ---
# Title: Uygulamanın adı (Jenerik)
app = FastAPI(title="Akıllı Ürün Tanıma Servisi")

# Veritabanı Klasörü
KLASOR = "urunler"

# --- GÖRÜNTÜ İŞLEME VE YAPAY ZEKA MOTORU ---
def goruntu_islee_ve_bul(gelen_resim_bytes):
    # 1. Gelen resmi (byte) OpenCV formatına çevir
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # 2. Ön İşleme (Griye Çevir)
    img1 = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    
    # 3. Kalite Artırma (CLAHE - Kontrast Dengeleme)
    # Bu özellik karanlık veya çok parlak fotoğrafları düzeltir
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    
    # 4. Keskinleştirme (Detayları Ortaya Çıkar)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img1 = cv2.filter2D(img1, -1, kernel)
    
    # 5. SIFT Algoritması (Parmak İzi Çıkarma)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    
    if des1 is None: return None, 0

    en_yuksek_skor = 0
    bulunan_kod = "Bulunamadı"

    # Veritabanını Tara
    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            # Veritabanındaki resmi oku
            db_path = os.path.join(KLASOR, dosya)
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE) # Direkt gri oku
            if db_img is None: continue
            
            # Veritabanı resmine de aynı iyileştirmeyi yap
            db_img = clahe.apply(db_img)
            
            kp2, des2 = sift.detectAndCompute(db_img, None)
            if des2 is None: continue
            
            # Eşleştirme (FLANN - Hızlı Tarama)
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Eleme (Lowe's Ratio Test)
            iyi_eslesmeler = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    iyi_eslesmeler.append(m)
            
            # Geometrik Doğrulama (RANSAC)
            # Rastgele benzerlikleri eler, sadece gerçek eşleşmeyi sayar
            if len(iyi_eslesmeler) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    skor = sum(mask.ravel().tolist())
                    # Eğer bu skor şimdiye kadarkilerin en iyisiyse, kaydet
                    if skor > en_yuksek_skor:
                        en_yuksek_skor = skor
                        bulunan_kod = dosya.split(".")[0] # Dosya adını (örn: 1704092) al
        except:
            continue

    return bulunan_kod, en_yuksek_skor

# --- İNTERNETE AÇILAN KAPI (API) ---
# Burası telefonun bağlanacağı kapıdır.

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mesaj": "Sunucu 7/24 Calisiyor"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    # 1. Telefondan gelen resmi oku
    resim_verisi = await file.read()
    
    # 2. İşle
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # 3. Sonuç Kararı (Eşik Değeri)
    ESIK = 6 
    
    if skor >= ESIK:
        return {
            "sonuc": True, 
            "urun_kodu": kod, 
            "guven_skoru": skor,
            "mesaj": "Ürün başarıyla eşleşti."
        }
    else:
        return {
            "sonuc": False, 
            "urun_kodu": None, 
            "guven_skoru": skor,
            "mesaj": "Eşleşme bulunamadı."
        }