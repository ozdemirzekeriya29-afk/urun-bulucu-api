from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import os
import json
import easyocr

app = FastAPI(title="Akıllı Ürün Bulucu - Full Versiyon")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"

# OCR Motoru (Yazı Okuma)
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

# --- YARDIMCI: Veritabanı Yükle ---
def veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

# --- 1. GÖRSEL ANALİZ (AKAZE - Şekil Tanıma) ---
def gorsel_puan_hesapla(aranan_resim, veritabani_klasor):
    # Resmi griye çevir
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    
    # AKAZE algoritmasını başlat
    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(img_gray, None)
    
    if des1 is None: return {} # Özellik bulunamadıysa çık

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    gorsel_skorlar = {} 

    if not os.path.exists(veritabani_klasor): return {}

    # Klasördeki tüm resimlerle karşılaştır
    for dosya in os.listdir(veritabani_klasor):
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        try:
            kod = dosya.split(".")[0] # dosya adı: "1704.jpg" -> kod: "1704"
            path = os.path.join(veritabani_klasor, dosya)
            
            # Veritabanı resmini oku
            db_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if db_img is None: continue

            # Boyut eşitleme (Hız için)
            h, w = db_img.shape[:2]
            if w > 600:
                s = 600/w
                db_img = cv2.resize(db_img, (600, int(h*s)))

            kp2, des2 = akaze.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                matches = bf.match(des1, des2)
                # Sadece çok iyi eşleşmeleri say
                iyi_eslesmeler = [m for m in matches if m.distance < 40]
                skor = len(iyi_eslesmeler)
                
                if skor > 0:
                    # Aynı koddan birden fazla resim varsa en iyisini al
                    if skor > gorsel_skorlar.get(kod, 0):
                        gorsel_skorlar[kod] = skor
        except: pass
        
    return gorsel_skorlar

# --- 2. YAZI ANALİZİ (OCR - Kelime Tanıma) ---
def yazi_puan_hesapla(resim, veritabani_json):
    try:
        # Resimdeki yazıları oku
        okunan_yazilar = reader.readtext(resim, detail=0)
        okunanlar = [yazi.lower() for yazi in okunan_yazilar]
        
        yazi_skorlar = {}

        for kod, detay in veritabani_json.items():
            anahtar_kelimeler = detay.get("anahtar_kelimeler", [])
            eslesme_sayisi = 0
            for anahtar in anahtar_kelimeler:
                for okunan in okunanlar:
                    if anahtar in okunan:
                        eslesme_sayisi += 1
            
            if eslesme_sayisi > 0:
                yazi_skorlar[kod] = eslesme_sayisi
            
        return yazi_skorlar
    except:
        return {}

# --- ANA MERKEZ: HEPSİNİ BİRLEŞTİR ---
def analiz_et(gelen_resim_bytes):
    veritabani_json = veritabani_yukle()
    
    # Byte verisini resme çevir
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if resim is None: return "Bulunamadı", 0, {}

    # Resmi biraz küçült (Hız için)
    h, w = resim.shape[:2]
    if w > 1000:
        s = 1000/w
        resim = cv2.resize(resim, (1000, int(h*s)))

    # 1. Görsel Skorları Al
    gorsel_skorlar = gorsel_puan_hesapla(resim, KLASOR)
    
    # 2. Yazı Skorları Al
    yazi_skorlar = yazi_puan_hesapla(resim, veritabani_json)
    
    # 3. Puanları Topla
    final_skorlar = []
    tum_kodlar = set(list(gorsel_skorlar.keys()) + list(yazi_skorlar.keys()))
    
    for kod in tum_kodlar:
        g_puan = gorsel_skorlar.get(kod, 0)
        y_puan = yazi_skorlar.get(kod, 0)
        
        # Formül: Görsel puan + (Yazı puanı * 20)
        # Yani 1 kelime tutması, 20 görsel nokta tutmasına bedeldir.
        toplam_puan = g_puan + (y_puan * 20)
        
        final_skorlar.append({"kod": kod, "toplam": toplam_puan})

    # En yüksek puanlıyı en üste al
    final_skorlar.sort(key=lambda x: x["toplam"], reverse=True)
    
    if not final_skorlar: return "Bulunamadı", 0, {}
    
    kazanan = final_skorlar[0]
    
    # Baraj Puanı (Hata yapmaması için en az bu kadar benzemeli)
    if kazanan["toplam"] >= 15:
        detay = veritabani_json.get(kazanan["kod"], {"ad": "Bilinmiyor"})
        return kazanan["kod"], kazanan["toplam"], detay
    else:
        return "Bulunamadı", kazanan["toplam"], {}

# --- ENDPOINT 1: RESİMLE ARAMA ---
@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor, detay = analiz_et(resim_verisi)
    
    if kod != "Bulunamadı":
        return {
            "sonuc": True, 
            "urun_kodu": kod, 
            "guven_skoru": skor, 
            "urun_detay": detay,
            "mesaj": "Tam Eşleşme"
        }
    else:
        return {"sonuc": False, "guven_skoru": skor, "mesaj": "Eşleşme Yok"}

# --- ENDPOINT 2: YAZI İLE (MANUEL) ARAMA ---
@app.post("/ara_metin")
async def metinle_ara(arama: str = Form(...)):
    print(f"Gelen Arama İsteği: {arama}")
    veritabani = veritabani_yukle()
    arama = arama.lower()
    
    for kod, detay in veritabani.items():
        anahtar_kelimeler = detay.get("anahtar_kelimeler", [])
        
        # Kullanıcının yazdığı kelime, anahtar kelimelerin içinde geçiyor mu?
        for kelime in anahtar_kelimeler:
            if arama in kelime or kelime in arama:
                return {
                    "sonuc": True,
                    "urun_kodu": kod,
                    "urun_detay": detay,
                    "mesaj": "Kelime eşleşmesi bulundu"
                }

    return {"sonuc": False, "mesaj": "Benzer ürün bulunamadı"}
