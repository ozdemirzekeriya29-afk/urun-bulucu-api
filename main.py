from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc
import json
import easyocr

app = FastAPI(title="Akıllı Ürün Tanıma - OTO KIRPMA V13")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

# --- SİHİRLİ FONKSİYON: OTOMATİK NESNE KIRPMA ---
def akilli_nesne_bul(img):
    """
    Fotoğraftaki en belirgin nesneyi (kutuyu) bulur ve kırpar.
    Google Lens efekti için arka planı temizler.
    """
    try:
        # Griye çevir ve bulanıklaştır (Gürültüyü sil)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kenarları bul (Canny Edge Detection)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Konturları (Çizgileri) bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return img # Hiçbir şey bulamazsa orijinali döndür
        
        # En büyük alanı kaplayan konturu bul (Muhtemelen ürün odur)
        c = max(contours, key=cv2.contourArea)
        
        # Dikdörtgen içine al
        x, y, w, h = cv2.boundingRect(c)
        
        # Eğer bulunan parça çok küçükse (Leke falansa) kırpma, orijinal kalsın
        h_img, w_img = img.shape[:2]
        if w * h < (w_img * h_img) * 0.10: # Resmin %10'undan küçükse yoksay
            return img
            
        # Kırpma işlemi (Biraz pay bırakarak - Padding)
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + pad*2)
        h = min(h_img - y, h + pad*2)
        
        kirpilmis = img[y:y+h, x:x+w]
        return kirpilmis
    except:
        return img # Hata olursa orijinali döndür

# --- STANDART FONKSİYONLAR ---
def urun_veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

def urun_bilgisi_getir(kod):
    if not os.path.exists(JSON_DOSYASI): return {"ad": kod, "fiyat": "?", "etiketler": []}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        veri = json.load(f)
    return veri.get(kod, {"ad": kod, "fiyat": "?", "etiketler": []})

def metin_ile_bul(okunan_metinler, veritabani):
    okunanlar = [kelime.lower() for kelime in okunan_metinler]
    en_iyi_kod = None
    en_cok_eslesme = 0
    for kod, detay in veritabani.items():
        etiketler = detay.get("etiketler", [])
        eslesme = 0
        for etiket in etiketler:
            for okunan in okunanlar:
                if etiket in okunan: eslesme += 1
        if eslesme > en_cok_eslesme:
            en_cok_eslesme = eslesme
            en_iyi_kod = kod
    if en_cok_eslesme >= 1: return en_iyi_kod, 100
    return None, 0

# --- ANA MOTOR ---
def goruntu_islee_ve_bul(gelen_resim_bytes):
    veritabani = urun_veritabani_yukle()
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    orijinal_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orijinal_resim is None: return None, 0, "Hata"

    # ADIM 1: OTO KIRPMA (Google Lens Efekti)
    # Resmi analiz etmeden önce gereksiz halıyı/masayı kes at.
    aranan_resim = akilli_nesne_bul(orijinal_resim)

    # ... (Buradan sonrası V12 ile aynı: AKAZE + OCR) ...
    # AKAZE KISMI
    try:
        # Boyutlandırma
        h, w = aranan_resim.shape[:2]
        if w > 1000:
            s = 1000/w
            aranan_resim = cv2.resize(aranan_resim, (1000, int(h*s)))

        img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
        akaze = cv2.AKAZE_create(threshold=0.001)
        kp1, des1 = akaze.detectAndCompute(img_gray, None)
        
        en_iyi_skor = 0
        en_iyi_kod = "Bulunamadı"

        if des1 is not None and os.path.exists(KLASOR):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            for dosya in os.listdir(KLASOR):
                if not dosya.endswith((".jpg", ".png")): continue
                try:
                    temel_kod = dosya.split("_")[0].split(".")[0]
                    db_path = os.path.join(KLASOR, dosya)
                    db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
                    if db_img is None: continue
                    h, w = db_img.shape[:2]
                    if w > 800:
                        s = 800/w
                        db_img = cv2.resize(db_img, (800, int(h*s)))
                    kp2, des2 = akaze.detectAndCompute(db_img, None)
                    if des2 is not None and len(des2) > 10:
                        matches = bf.match(des1, des2)
                        iyi = [m for m in matches if m.distance < 60]
                        if len(iyi) >= 12:
                            if len(iyi) > en_iyi_skor:
                                en_iyi_skor = len(iyi)
                                en_iyi_kod = temel_kod
                except: pass
        
        if en_iyi_skor > 20: return en_iyi_kod, en_iyi_skor, "Görselden (Oto Kırpma)"
    except: pass

    # OCR KISMI (Yazı Okuma)
    print("OCR deneniyor...")
    try:
        sonuclar = reader.readtext(aranan_resim, detail=0)
        ocr_kod, ocr_skor = metin_ile_bul(sonuclar, veritabani)
        if ocr_kod: return ocr_kod, 100, "Yazıdan (OCR)"
    except: pass

    return en_iyi_kod, en_iyi_skor, "Görsel"

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor, kaynak = goruntu_islee_ve_bul(resim_verisi)
    detaylar = urun_bilgisi_getir(kod)
    
    if skor >= 15 and kod != "Bulunamadı":
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "urun_detay": detaylar}
    else:
        return {"sonuc": False, "mesaj": "Eşleşme Yok"}
        
