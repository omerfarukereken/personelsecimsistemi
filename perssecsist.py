import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ClassificationReport
from xgboost import XGBClassifier
import datetime
import warnings
import pickle
warnings.filterwarnings('ignore')



def anaprogram():

    secim = st.sidebar.selectbox("MENÜ", ("Başvuru Girişi", "Yönetici Girişi", "İletişim"), key="a")


    if secim == "Başvuru Girişi":
        st.title("PERSONEL BAŞVURU SİSTEMİ")

        cift_engelleyici = 0
        st.subheader("Başvuru Girişi")
        left, right = st.beta_columns(2)
        with left:
            isim = st.text_input("İsim")
        with right:
            soyisim = st.text_input("Soyisim")

        dogumtarihi = st.date_input("Doğum Tarihiniz", min_value=datetime.date(1980, 1, 1), max_value=datetime.date(2000, 1, 1))
        def yas_hesapla(dogumtarihi):
            bugün = datetime.date.today()
            return bugün.year - dogumtarihi.year - ((bugün.month, bugün.day) < (dogumtarihi.month, dogumtarihi.day))
        yas = yas_hesapla(dogumtarihi)
        ###################################################
        ikametili= st.text_input("İkamet İli")
        ikametilcesi= st.text_input("İkamet İlçesi")
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Cinsiyet")
        with b:
            erkek = st.checkbox('Erkek')
        with c:
            kız = st.checkbox('Kız')
        if erkek+kız>1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Medenidurum")
        with b:
            medenidurum_evli = st.checkbox('Evli')
        with c:
            medenidurum_bekar = st.checkbox('Bekar')
        if medenidurum_evli+medenidurum_bekar>1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Askerlik Durumu")
        with b:
            askerlikdurumu_tamam = st.checkbox('Muaf/Tamamlandı')
        with c:
            askerlikdurumu_tamamlanmadı = st.checkbox('Tamamlanmadı')
        if askerlikdurumu_tamam+askerlikdurumu_tamamlanmadı>1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        st.write("Eğitim Durumunuz")
        a, b, c, d = st.beta_columns(4)
        with a:
            egitim_önlisans = st.checkbox('Önlisans')
        with b:
            egitim_lisans = st.checkbox('Lisans')
        with c:
            egitim_yükseklisans = st.checkbox('Yükseklisans')
        with d:
            egitim_doktora = st.checkbox('Doktora')
        if egitim_önlisans+egitim_lisans+egitim_yükseklisans+egitim_doktora>1:
            st.warning("Lütfen Son Mezuniyetinize Göre Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        lisansbölümü_Bilgisayar = 0
        lisansbölümü_Bilgisayar_Mühendisliği = 0
        lisansbölümü_Endüstri_Mühendisliği = 0
        lisansbölümü_Matematik = 0
        lisansbölümü_Matematik_Mühendisliği = 0
        lisansbölümü_İstatistik = 0
        yükseklisansbölümü_Bilgisayar_Mühendisliği = 0
        yükseklisansbölümü_Endüstri_Mühendisliği = 0
        yükseklisansbölümü_Veri_Bilimi = 0
        yükseklisansbölümü_Yönetim_Bilişim_Sistemleri = 0
        yükseklisansbölümü_İstatistik = 0
        doktorabölümü_Bilgisayar_Mühendisliği = 0
        doktorabölümü_Endüstri_Mühendisliği = 0
        doktorabölümü_Veri_Bilimi = 0
        doktorabölümü_Yönetim_Bilişim_Sistemleri = 0
        doktorabölümü_İstatistik = 0
        lisans_diger = 0
        yükseklisans_diger = 0
        doktorabölümü_diger = 0
        lisansokul = 0
        yükseklisansokul = 0
        doktoraokul = 0

        if egitim_önlisans==1 or egitim_lisans==1 or egitim_yükseklisans==1 or egitim_doktora==1:
            lisansbölümü = st.selectbox("Lisans/Önlisans Bölümü", ("Bilgisayar Önlisans", "Bilgisayar Mühendisliği",
            "Endüstri Mühendisliği", "Matematik", "Matematik Mühendisliği", "Diğer"), key="lsbl")
            lisansokul = st.text_input('Lütfen Lisans Eğitiminizi Tamamladığınız Okulu Yazınız:')
            if lisansbölümü == "Bilgisayar":
                lisansbölümü_Bilgisayar=1
            elif lisansbölümü == "Bilgisayar Mühendisliği":
                lisansbölümü_Bilgisayar_Mühendisliği=1
            elif lisansbölümü == "Endüstri Mühendisliği":
                lisansbölümü_Endüstri_Mühendisliği=1
            elif lisansbölümü == "Matematik":
                lisansbölümü_Matematik=1
            elif lisansbölümü == "Matematik Mühendisliği":
                lisansbölümü_Matematik_Mühendisliği=1
            elif lisansbölümü == "Diğer":
                lisans_diger = st.text_input('Lütfen Lisans Bölümünüzü Belirtiniz:')

        if egitim_yükseklisans==1 or egitim_doktora==1:
            yükseklisansbölümü = st.selectbox("Yüksek Lisans Bölümü", ("Bilgisayar Mühendisliği", "Endüstri Mühendisliği",
            "Veri Bilimi", "Yönetim Bilişim Sistemleri", "İstatistik", "Diğer"), key="ylbl")
            yükseklisansokul = st.text_input('Lütfen Yüksek Lisans Eğitiminizi Tamamladığınız Okulu Yazınız:')
            if yükseklisansbölümü == "Bilgisayar_Mühendisliği":
                yükseklisansbölümü_Bilgisayar_Mühendisliği=1
            elif yükseklisansbölümü == "Endüstri Mühendisliği":
                yükseklisansbölümü_Endüstri_Mühendisliği=1
            elif yükseklisansbölümü == "Veri Bilimi":
                yükseklisansbölümü_Veri_Bilimi=1
            elif yükseklisansbölümü == "Yönetim Bilişim Sistemleri":
                yükseklisansbölümü_Yönetim_Bilişim_Sistemleri=1
            elif yükseklisansbölümü == "İstatistik":
                yükseklisansbölümü_İstatistik=1
            elif yükseklisansbölümü == "Diğer":
                yükseklisans_diger = st.text_input('Lütfen Yüksek Lisans Bölümünüzü Belirtiniz:')

        if egitim_doktora==1:
            doktorabölümü = st.selectbox("Doktora Bölümü", ("Bilgisayar Mühendisliği", "Endüstri Mühendisliği",
            "Veri Bilimi", "Yönetim Bilişim Sistemleri", "İstatistik", "Diğer"), key="dkbl")
            doktoraokul = st.text_input('Lütfen Doktora Eğitiminizi Tamamladığınız Okulu Yazınız:')
            if doktorabölümü == "Bilgisayar_Mühendisliği":
                doktorabölümü_Bilgisayar_Mühendisliği=1
            elif doktorabölümü == "Endüstri Mühendisliği":
                doktorabölümü_Endüstri_Mühendisliği=1
            elif doktorabölümü == "Veri Bilimi":
                doktorabölümü_Veri_Bilimi=1
            elif doktorabölümü == "Yönetim Bilişim Sistemleri":
                doktorabölümü_Yönetim_Bilişim_Sistemleri=1
            elif doktorabölümü == "İstatistik":
                doktorabölümü_İstatistik=1
            elif doktorabölümü == "Diğer":
                doktorabölümü_diger = st.text_input('Lütfen Doktora Bölümünüzü Belirtiniz:')
        ###################################################
        diger_prog_dil = 0
        st.write("Bildiğiniz Programlama Dillerini İşaretleyiniz")
        a, b, c, d, e = st.beta_columns(5)
        with a:
            program_python = st.checkbox('Python')
        with b:
            program_R = st.checkbox('R')
        with c:
            program_C = st.checkbox('C')
        with d:
            program_Cplus = st.checkbox('C++')
        with e:
            program_sql = st.checkbox('SQL')
        a, b, c, d, e = st.beta_columns(5)
        with a:
            program_javascript = st.checkbox('Javascript')
        with b:
            program_scala = st.checkbox('Scala')
        with c:
            program_julia = st.checkbox('Julia')
        with d:
            program_PHP = st.checkbox('PHP')
        with e:
            diger = st.checkbox('Diğer', key="dig_prog")
        if diger == 1:
            diger_prog_dil = st.text_input('Lütfen Bildiğiniz Diğer Dilleri Belirtiniz:')
        ###################################################
        st.write("2018-2020 Yılları Arasında Şirketimizin Düzenlediği Yapay Zeka Kursuna Katıldınız mı?")
        a, b, c = st.beta_columns(3)
        with a:
            q = 0
        with b:
            kurs_aldım = st.checkbox('Katıldım')
        with c:
            kurs_almadım = st.checkbox('Katılmadım')
        ###################################################
        yzbilgi = st.slider("Yapay Zeka Bilginizi 0-4 Arasında Puanlayınız", 0, 4)
        ###################################################
        diger_yabancı_dil = 0
        st.write("Bildiğiniz Yabancı Dilleri İşaretleyiniz")
        a, b, c, d, e = st.beta_columns(5)
        with a:
            dil_ingilizce = st.checkbox('İngilizce')
        with b:
            dil_almanca = st.checkbox('Almanca')
        with c:
            dil_fransızca = st.checkbox('Fransızca')
        with d:
            dil_arapça = st.checkbox('Arapça')
        with e:
            dil_rusça = st.checkbox('Rusça')
        a, b = st.beta_columns(2)
        with a:
            dil_bilmiyor = st.checkbox('Yabancı Dil Bilmiyorum')
        with b:
            diger_yb_dil = st.checkbox('Diğer', key="dig_dil")
        if diger_yb_dil == 1:
            diger_yabancı_dil = st.text_input('Lütfen Bildiğiniz Diğer Dilleri Belirtiniz:')
        ###################################################
        alantecrübesi= st.text_input("IT Alanındaki İş Tecrübeniz Kaç Yıldır?")
        maaşbeklentisi= st.text_input("Maaş Beklentiniz Aylık Kaç TL'dir?")
        ensonücret= st.text_input("En Son Çalıştığınız Şirketteki Maaşınız Aylık Kaç TL'ydi?")
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Adli Sicil Kaydı")
        with b:
            adlisicilkaydı_yoktur = st.checkbox('Adli Sicil Kaydım Yoktur')
        with c:
            adlisicilkaydı_vardır = st.checkbox('Adli Sicil Kaydım Vardır')
        if adlisicilkaydı_yoktur + adlisicilkaydı_vardır > 1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Seyahat Engel Durumu")
        with b:
            seyahatengel_yoktur = st.checkbox('Seyahat Engelim Yoktur')
        with c:
            seyahatengel_vardır = st.checkbox('Seyahat Engelim Vardır')
        if seyahatengel_yoktur + seyahatengel_vardır > 1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Referans")
        with b:
            referans_yoktur = st.checkbox('Referansım Yoktur')
        with c:
            referans_vardır = st.checkbox('Referansım Vardır')
        if referans_yoktur+referans_vardır>1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1
        ###################################################
        a, b, c = st.beta_columns(3)
        with a:
            st.write("Vardiyalı Çalışma Durumunuz?")
        with b:
            vardiyaimkanı_çalışabilirim = st.checkbox('Çalışabilirim')
        with c:
            vardiyaimkanı_çalışamam = st.checkbox('Çalışamam')
        if vardiyaimkanı_çalışabilirim+vardiyaimkanı_çalışamam>1:
            st.warning("Lütfen Tek Kutucuğu Seçiniz")
            cift_engelleyici = 1

        model = pickle.load(open("my_model", "rb"))

        a, b, c, d = st.beta_columns(4)
        with a:
            q=1
        with b:
            q=1
        with c:
            q=1
        with d:
            if st.button("Tamamla") and cift_engelleyici!=1:
                yz_haric = {"isim":isim, "soyisim":soyisim, "dogumtarihi":dogumtarihi,"ikametili":ikametili,
                            "ikametilcesi":ikametilcesi, "cinsiyet_erkek":erkek, "cinsiyet_kız":kız,
                            "medenidurum_evli":medenidurum_evli, "medenidurum_bekar":medenidurum_bekar,
                            "askerlikdurumu_tamam":askerlikdurumu_tamam, "askerlikdurumu_tamamlanmadı":askerlikdurumu_tamamlanmadı,
                            "lisans_diger":lisans_diger,"yükseklisans_diger":yükseklisans_diger, "doktorabölümü_diger":doktorabölümü_diger,
                            "diger_prog_dil":diger_prog_dil, "diger_yabancı_dil":diger_yabancı_dil, "lisansokul":lisansokul,
                            "yükseklisansokul":yükseklisansokul, "doktoraokul":doktoraokul}
                listem = {'yas': yas, 'alantecrübesi': alantecrübesi, 'maaşbeklentisi': maaşbeklentisi,
                          'ensonücret': ensonücret, 'yzbilgi': yzbilgi,
                          'egitim_doktora': egitim_doktora, 'egitim_lisans': egitim_lisans,
                          'egitim_yükseklisans': egitim_yükseklisans, 'egitim_önlisans': egitim_önlisans,
                          'lisansbölümü_Bilgisayar': lisansbölümü_Bilgisayar,
                          'lisansbölümü_Bilgisayar Mühendisliği': lisansbölümü_Bilgisayar_Mühendisliği,
                          'lisansbölümü_Endüstri Mühendisliği': lisansbölümü_Endüstri_Mühendisliği,
                          'lisansbölümü_Matematik': lisansbölümü_Matematik,
                          'lisansbölümü_Matematik Mühendisliği': lisansbölümü_Matematik_Mühendisliği,
                          'lisansbölümü_İstatistik': lisansbölümü_İstatistik,
                          'yükseklisansbölümü_Bilgisayar Mühendisliği': yükseklisansbölümü_Bilgisayar_Mühendisliği,
                          'yükseklisansbölümü_Endüstri Mühendisliği': yükseklisansbölümü_Endüstri_Mühendisliği,
                          'yükseklisansbölümü_Veri Bilimi': yükseklisansbölümü_Veri_Bilimi,
                          'yükseklisansbölümü_Yönetim Bilişim Sistemleri': yükseklisansbölümü_Yönetim_Bilişim_Sistemleri,
                          'yükseklisansbölümü_İstatistik': yükseklisansbölümü_İstatistik,
                          'doktorabölümü_Bilgisayar Mühendisliği': doktorabölümü_Bilgisayar_Mühendisliği,
                          'doktorabölümü_Endüstri Mühendisliği': doktorabölümü_Endüstri_Mühendisliği,
                          'doktorabölümü_Veri Bilimi': doktorabölümü_Veri_Bilimi,
                          'doktorabölümü_Yönetim Bilişim Sistemleri': doktorabölümü_Yönetim_Bilişim_Sistemleri,
                          'doktorabölümü_İstatistik': doktorabölümü_İstatistik,
                          'adlisicilkaydı_vardır': adlisicilkaydı_vardır,
                          'adlisicilkaydı_yoktur': adlisicilkaydı_yoktur, 'seyahatengeli_vardır': seyahatengel_vardır,
                          'seyahatengeli_yoktur': seyahatengel_yoktur, 'referans_vardır': referans_vardır,
                          'referans_yoktur': referans_yoktur, 'kurs_aldım': kurs_aldım,
                          'kurs_almadım': kurs_almadım, 'vardiyaimkanı_çalışabilirim': vardiyaimkanı_çalışabilirim,
                          'vardiyaimkanı_çalışamam': vardiyaimkanı_çalışamam, 'dil_almanca': dil_almanca,
                          'dil_arapça': dil_arapça, 'dil_bilmiyor': dil_bilmiyor, 'dil_fransızca': dil_fransızca,
                          'dil_ingilizce': dil_ingilizce, 'dil_rusça': dil_rusça, 'program_C': program_C,
                          'program_C++': program_Cplus, 'program_PHP': program_PHP, 'program_R': program_R,
                          'program_javascript': program_javascript, 'program_julia': program_julia,
                          'program_python': program_python, 'program_scala': program_scala, 'program_sql': program_sql}
                yz_haric_verisi = pd.DataFrame.from_dict(yz_haric, orient="index").T
                yz_verisi = pd.DataFrame.from_dict(listem, orient="index").T
                tum_veriler = yz_haric_verisi.join(yz_verisi)

                try:
                    basvurular = pd.read_csv("basvurular_yz.csv")
                    basvurular = basvurular.append(yz_verisi, ignore_index=True)
                    tum_veri = pd.read_csv("basvurular_tum_veri.csv")
                    tum_veri = tum_veri.append(tum_veriler, ignore_index=True)
                except:
                    basvurular = yz_verisi
                    tum_veri = tum_veriler

                basvurular.to_csv("basvurular_yz.csv",  index=False)
                tum_veri.to_csv("basvurular_tum_veri.csv",  index=False)
                st.write("BAŞVURUNUZ TAMAMLANMIŞTIR")
                pred = model.predict(yz_verisi)
                if pred==1:
                    st.write("MÜLAKATA ÇAĞIRILACAK")
                else:
                    st.write("BAŞARISIZ")
            elif cift_engelleyici == 1:
                st.warning("LÜTFEN BİLGİLERİNİZİ KONTROL EDİNİZ")
        ###################################################

    elif secim == "Yönetici Girişi":
        st.title("PERSONEL SEÇİM SİSTEMİ")


        sifre = st.sidebar.text_input("Parola", type="password", key="b")

        if st.sidebar.checkbox("Giriş"):
            if sifre == "":
                islem = st.sidebar.selectbox("İşlemi Seçiniz", ("Model Puanlamalarını Görme", "Başvuru Değerlendirme"), key="c")
                st.header(islem)
                secilen_model = st.sidebar.selectbox("Modeli Seçiniz",
                ("Lojistik Regresyon", "K En Yakın Komşular", "Karar Ağacı", "Rastgele Orman", "XGBoost"), key="d")
                def veri_cek(islem):
                    if islem == "Model Puanlamalarını Görme":
                        df = pd.read_csv("yz.csv")
                        X = df.drop("karar", axis=1)
                        y = df["karar"]
                        return X, y
                    elif islem == "Başvuru Değerlendirme":
                        df = pd.read_csv("yz.csv")
                        X = df.drop("karar", axis=1)
                        y = df["karar"]
                        basvuru_degerlendirme = pd.read_csv("basvurular_yz.csv")
                        return X, y, basvuru_degerlendirme

                if islem == "Model Puanlamalarını Görme":
                    X, y = veri_cek(islem)
                elif islem == "Başvuru Değerlendirme":
                    X, y, basvuru_degerlendirme = veri_cek(islem)

                def parametre_belirle(secilen_model):
                    param = dict()
                    if secilen_model == "K En Yakın Komşular":
                        K = st.sidebar.slider("K", 1, 15)
                        param["K"] = K
                    elif secilen_model == "Rastgele Orman":
                        max_depth = st.sidebar.slider("max_depth", 2, 15)
                        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
                        param["max_depth"] = max_depth
                        param["n_estimators"] = n_estimators
                    return param

                parametreler = parametre_belirle(secilen_model)

                def modeli_calıstır(secilen_model, parametreler):
                    if secilen_model == "Lojistik Regresyon":
                        modelim = LogisticRegression()
                    elif secilen_model == "K En Yakın Komşular":
                        modelim = KNeighborsClassifier(n_neighbors=parametreler["K"])
                    elif secilen_model == "Karar Ağacı":
                        modelim = DecisionTreeClassifier()
                    elif secilen_model == "Rastgele Orman":
                        modelim = RandomForestClassifier(max_depth=parametreler["max_depth"],
                        n_estimators = parametreler["n_estimators"], random_state=42)
                    elif secilen_model == "XGBoost":
                        modelim = XGBClassifier()
                    return modelim

                model = modeli_calıstır(secilen_model, parametreler)

                X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

                if secilen_model=="Lojistik Regresyon" or secilen_model == "K En Yakın Komşular":
                    normalize = StandardScaler()
                    X_egitim = normalize.fit_transform(X_egitim)
                    X_test = normalize.transform(X_test)
                    X = normalize.transform(X)

                if islem == "Model Puanlamalarını Görme":
                    model.fit(X_egitim, y_egitim)
                    y_tahmin = model.predict(X_test)
                    doğruluk_oran = accuracy_score(y_test, y_tahmin)
                    st.markdown(f"**Model = **{secilen_model}")
                    st.write(f"**Doğruluk Oranı = **{doğruluk_oran}")
                    st.subheader("Sınıflandırma Raporu")
                    fig1= plt.figure()
                    viz = ClassificationReport(model, support=True)
                    viz.fit(X_egitim, y_egitim)
                    viz.score(X_test, y_test)
                    plt.xticks([0.5, 1.5, 2.5, 3.5], ['Hassasiyet', 'Geri Çağırma', 'F1', 'Sayı'])
                    plt.yticks([0.5, 1.5], ['0', '1'])
                    st.pyplot(fig1)
                    cnf_matrix = confusion_matrix(y_test, y_tahmin)
                    fig2=plt.figure()
                    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu",fmt='d')
                    plt.ylabel('Asıl Etiket')
                    plt.xlabel('Tahmin Edilen Etiket')
                    st.pyplot(fig2)

                elif islem == "Başvuru Değerlendirme":
                    model.fit(X, y)
                    y_pred = model.predict(basvuru_degerlendirme)
                    cıktı = pd.read_csv("basvurular_tum_veri.csv")
                    for i in range(len(cıktı)):
                        sonuc = pd.DataFrame(cıktı.loc[i])
                        sonuc = sonuc.T
                        sonuc = sonuc[sonuc != False]
                        sonuc = sonuc.dropna(axis=1)
                        sonuc.insert(0, "karar", y_pred[i])
                        st.write(sonuc)
            else:
                st.write("""
                YANLIŞ PAROLA !!!
                """)
    elif secim == "İletişim":
        st.title("ŞİRKETİMİZE AİT İLETİŞİM BİLGİLERİ")
        st.markdown("Şirketimize ulaşmak ve ekstra bilgi edinmek için aşağıdaki " 
                    "iletişim yöntemleriyle müşteri temsilcimizle görüşebilirsiniz.")
        st.markdown("**Telefon Numaramız :** 02120000000")
        st.markdown("**Faks :** 02120000000")
        st.markdown("**Email :** bilişim@bilişim.com")
        st.markdown("**Çalışma Saatlerimiz** 09:00-18:00")



        st.markdown("**Adres :** Levent Mah. Deniz Cad. Bilişim Plaza No:1 Kağıthane/İstanbul")

if __name__ == '__main__':
    anaprogram()




