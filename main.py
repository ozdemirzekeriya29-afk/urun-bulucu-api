from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc
import json
import easyocr

app = FastAPI(title="Akıllı Ürün Tanıma - GÖRSEL + YAZI (HİBRİT)")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"

# OCR Motoru (GPU yoksa cpu modunda çalışır)
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

# --- YARDIMCI: Veritabanı Yükle ---
def veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

# --- MOTOR 1: GÖRSEL ANALİZ (AKAZE) ---
def gorsel_puan_hesapla(aranan_resim, veritabani_klasor):
    """
    Resmin şekline bakar ve görsel benzerlik puanı verir.
    """
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(img_gray, None)
    
    if des1 is None: return {}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    gorsel_skorlar = {} # { "1704": 25, "1705": 10 }

    if not os.path.exists(veritabani_klasor): return {}

    for dosya in os.listdir(veritabani_klasor):
        if not dosya.endswith((".jpg", ".png")): continue
        try:
            # Dosya adından kodu al (1704_on.jpg -> 1704)
            kod = dosya.split("_")[0].split(".")[0]
            
            path = os.path.join(veritabani_klasor, dosya)
            db_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Hız için küçült
            h, w = db_img.shape[:2]
            if w > 600:
                s = 600/w
                db_img = cv2.resize(db_img, (600, int(h*s)))

            kp2, des2 = akaze.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                matches = bf.match(des1, des2)
                iyi_eslesmeler = [m for m in matches if m.distance < 60]
                skor = len(iyi_eslesmeler)
                
                # Eğer bu kod için daha önce düşük skor varsa güncelle
                if skor > gorsel_skorlar.get(kod, 0):
                    gorsel_skorlar[kod] = skor
        except: pass
        
    return gorsel_skorlar

# --- MOTOR 2: YAZI ANALİZİ (OCR) ---
def yazi_puan_hesapla(resim, veritabani_json):
    """
    Resimdeki yazıları okur ve anahtar kelimelerle eşleştirir.
    """
    try:
        # EasyOCR ile oku (Liste döner: ['ARBELLA', 'Burgu', '500g'])
        okunan_yazilar = reader.readtext(resim, detail=0)
        # Hepsini küçük harfe çevir
        okunanlar = [yazi.lower() for yazi in okunan_yazilar]
        
        yazi_skorlar = {} # { "1704": 2, "1705": 1 }

        for kod, detay in veritabani_json.items():
            anahtar_kelimeler = detay.get("anahtar_kelimeler", [])
            eslesme_sayisi = 0
            
            for anahtar in anahtar_kelimeler:
                # Okunan metinlerin içinde anahtar kelime geçiyor mu?
                # Örn: "arbella" kelimesi "arbella makarna" içinde var mı?
                for okunan in okunanlar:
                    if anahtar in okunan:
                        eslesme_sayisi += 1
                        break # Aynı kelimeyi 2 kere sayma
            
            yazi_skorlar[kod] = eslesme_sayisi
            
        return yazi_skorlar, okunan_yazilar
    except:
        return {}, []

# --- ANA MERKEZ ---
def analiz_et(gelen_resim_bytes):
    veritabani_json = veritabani_yukle()
    
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if resim is None: return None, 0, {}

    # Resmi Optimize Et (1000px)
    h, w = resim.shape[:2]
    if w > 1000:
        s = 1000/w
        resim = cv2.resize(resim, (1000, int(h*s)))

    # 1. GÖRSEL PUANLARI AL (AKAZE)
    gorsel_skorlar = gorsel_puan_hesapla(resim, KLASOR)
    
    # 2. YAZI PUANLARI AL (OCR)
    yazi_skorlar, okunanlar = yazi_puan_hesapla(resim, veritabani_json)
    
    # 3. HİBRİT PUANLAMA (BÜYÜK FİNAL)
    # Formül: (Görsel Puan) + (Yazı Puanı x 20)
    # Neden 20? Çünkü 1 kelime (örn: Burgu) görseldeki 20 noktaya bedeldir.
    
    final_skorlar = []
    
    # Tüm ürün kodlarını topla
    tum_kodlar = set(list(gorsel_skorlar.keys()) + list(yazi_skorlar.keys()))
    
    for kod in tum_kodlar:
        g_puan = gorsel_skorlar.get(kod, 0)
        y_puan = yazi_skorlar.get(kod, 0)
        
        # HİBRİT PUAN FORMÜLÜ
        toplam_puan = g_puan + (y_puan * 25) 
        
        final_skorlar.append({
            "kod": kod,
            "toplam": toplam_puan,
            "gorsel_detay": g_puan,
            "yazi_detay": y_puan
        })

    # Puanlara göre sırala
    final_skorlar.sort(key=lambda x: x["toplam"], reverse=True)
    
    if not final_skorlar: return "Bulunamadı", 0, {}
    
    kazanan = final_skorlar[0]
    kazanan_kod = kazanan["kod"]
    kazanan_skor = kazanan["toplam"]
    
    # --- KARAR MEKANİZMASI ---
    # Kazanması için barajı geçmeli
    # Baraj: Ya görseli çok iyi olacak (min 15)
    # Ya da en az 1 kelime okumuş olacak (25 puan)
    
    BARAJ = 25 
    
    if kazanan_skor >= BARAJ:
        detay = veritabani_json.get(kazanan_kod, {"ad": kazanan_kod, "fiyat": "?"})
        return kazanan_kod, kazanan_skor, detay
    else:
        return "Bulunamadı", kazanan_skor, {}

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
