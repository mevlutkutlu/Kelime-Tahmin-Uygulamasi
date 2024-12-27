import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        # Tarayıcıyı başlatır
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:8501")
        self.driver.maximize_window()
        time.sleep(3)  # Sayfanın yüklenmesini bekler

    def test_menu_navigation(self):
        driver = self.driver

        # Menüden "Yeni Kişi Ekle" seçeneğini seçer.
        yeni_kisi_menu = driver.find_element(By.XPATH, "//div[@role='radiogroup']//p[text()='Yeni Kişi Ekle']")
        yeni_kisi_menu.click()
        time.sleep(2)

        # Başlık kontrolü
        page_header = driver.find_element(By.TAG_NAME, "h2")
        self.assertIn("Yeni Kişi Ekle", page_header.text)

             
    def test_add_new_person(self):
        driver = self.driver

        # "Yeni Kişi Ekle" menüsüne tıklar.
        yeni_kisi_menu = driver.find_element(By.XPATH, "//div[@role='radiogroup']//p[text()='Yeni Kişi Ekle']")
        yeni_kisi_menu.click()
        time.sleep(2)

        kisi_adi_input = driver.find_element(By.XPATH, "//input[@aria-label='Yeni kişinin adını girin:']")
        kisi_adi_input.send_keys("TestKisi")
        kisi_adi_input.send_keys(Keys.ENTER)

        # Başarı mesajını bekler.
        success_message = WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, "//p[text()='MFCC özellikleri başarıyla oluşturuldu.']"))
        )
        print("Test başarıyla tamamlandı: ", success_message.text)
  
        # Tarayıcıyı kapatır.
        driver.quit() 
      
    
    def test_start_live_test(self):
        driver = self.driver


        # "Canlı Test Yap" menüsüne tıklar.
        yeni_kisi_menu = driver.find_element(By.XPATH, "//div[@role='radiogroup']//p[text()='Canlı Test Yap']")
        yeni_kisi_menu.click()
        time.sleep(2)


        # "Test Başlat" butonuna tıklama
        test_baslat_button = driver.find_element(By.XPATH, "//button[.//p[text()='Test Başlat']]")
        test_baslat_button.click()

       # Başarı mesajını bekler
        success_message = WebDriverWait(driver, 100).until(
        EC.presence_of_element_located((By.XPATH, "//p[text()='Test sonuçlanmıştır']"))
        )
        print("Test başarıyla tamamlandı: ", success_message.text)
    
    def tearDown(self):
        self.driver.quit()  # Testten sonra tarayıcıyı kapatır

if __name__ == "__main__":
    unittest.main()
