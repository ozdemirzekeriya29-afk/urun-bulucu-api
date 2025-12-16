from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc
import json
import easyocr

app = FastAPI(title="Akıllı Ürün Tanıma - FİYATSIZ MOD")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"

# OCR Motoru
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

# --- YARDIMCI: Veritabanı Yükle ---
def veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

# --- FİYATSIZ BİLGİ GETİRME ---
def urun_bilgisi_getir(kod):
    if not os.path.exists(JSON_DOSYASI):
        return {"ad": "Bilinmeyen Ürün"}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        veri = json.load(f)
    # Artık sadece AD dönüyor, fiyat yok.
    return veri.get(kod, {"ad": kod})

# --- GÖRSEL ANALİZ (AKAZE) ---
def gorsel_puan_hesapla(aranan_resim, veritabani_klasor):
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(img_gray, None)
    
    if des1 is None: return {}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    gorsel_skorlar = {} 

    if not os.path.exists(veritabani_klasor): return {}

    for dosya in os.listdir(veritabani_klasor):
        if not dosya.endswith((".jpg", ".png")): continue
        try:
            kod = dosya.split("_")[0].split(".")[0]
            path = os.path.join(veritabani_klasor, dosya)
            db_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            h, w = db_img.shape[:2]
            if w > 600:
                s = 600/w
                db_img = cv2.resize(db_img, (600, int(h*s)))

            kp2, des2 = akaze.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                matches = bf.match(des1, des2)
                iyi_eslesmeler = [m for m in matches if m.distance < 60]
                skor = len(iyi_eslesmeler)
                
                if skor > gorsel_skorlar.get(kod, 0):
                    gorsel_skorlar[kod] = skor
        except: pass
        
    return gorsel_skorlar

# --- YAZI ANALİZİ (OCR) ---
def yazi_puan_hesapla(resim, veritabani_json):
    try:
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
                        break 
            yazi_skorlar[kod] = eslesme_sayisi
            
        return yazi_skorlar
    except:
        return {}

# --- ANA MERKEZ ---
def analiz_et(gelen_resim_bytes):
    veritabani_json = veritabani_yukle()
    
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if resim is None: return None, 0, {}

    h, w = resim.shape[:2]
    if w > 1000:
        s = 1000/w
        resim = cv2.resize(resim, (1000, int(h*s)))

    gorsel_skorlar = gorsel_puan_hesapla(resim, KLASOR)
    yazi_skorlar = yazi_puan_hesapla(resim, veritabani_json)
    
    final_skorlar = []
    tum_kodlar = set(list(gorsel_skorlar.keys()) + list(yazi_skorlar.keys()))
    
    for kod in tum_kodlar:
        g_puan = gorsel_skorlar.get(kod, 0)
        y_puan = yazi_skorlar.get(kod, 0)
        toplam_puan = g_puan + (y_puan * 25) 
        
        final_skorlar.append({"kod": kod, "toplam": toplam_puan})

    final_skorlar.sort(key=lambda x: x["toplam"], reverse=True)
    
    if not final_skorlar: return "Bulunamadı", 0, {}
    
    kazanan = final_skorlar[0]
    
    # Baraj
    if kazanan["toplam"] >= 25:
        # Burada artık sadece AD bilgisini alıyoruz
        detay = urun_bilgisi_getir(kazanan["kod"])
        return kazanan["kod"], kazanan["toplam"], detay
    else:
        return "Bulunamadı", kazanan["toplam"], {}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor, detay = analiz_et(resim_verisi)
    
    if kod != "Bulunamadı":
        return {
            "sonuc": True, 
            "urun_kodu": kod, 
            "guven_skoru": skor, 
            "urun_detay": detay, # İçinde artık fiyat yok
            "mesaj": "Tam Eşleşme"
        }
    else:
        return {"sonuc": False, "guven_skoru": skor, "mesaj": "Eşleşme Yok"}
