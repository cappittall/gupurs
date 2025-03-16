SSH Anahtarı Oluşturma ve Saklama
Jetson Nano'da "gupurs" adında bir SSH anahtarı oluşturmak ve bunu SSH klasörüne saklamak için aşağıdaki adımları izleyebilirsiniz:

1. SSH Anahtarı Oluşturma
SSH anahtarı oluşturmak için aşağıdaki komutu kullanabilirsiniz:

ssh-keygen -t ed25519 -f ~/.ssh/gupurs -C "gupurs key for Jetson Nano"
Bu komut, Ed25519 algoritması kullanarak "gupurs" adında bir SSH anahtarı oluşturacaktır. Komut çalıştığında sizden bir parola girmenizi isteyebilir. ancak boş bırakarak da devam edebilirsiniz.

2. SSH Klasörünün İzinlerini Ayarlama
SSH klasörünün ve içindeki dosyaların doğru izinlere sahip olduğundan emin olalım:

chmod 700 ~/.ssh
chmod 600 ~/.ssh/gupurs
chmod 644 ~/.ssh/gupurs.pub

3. SSH Yapılandırma Dosyasına Ekleme
SSH anahtarını GitHub için yapılandırmak isterseniz, SSH yapılandırma dosyasına ekleyebilirsiniz:

touch ~/.ssh/config
chmod 600 ~/.ssh/config
Ardından, config dosyasını düzenleyerek aşağıdaki içeriği ekleyebilirsiniz:

echo 'Host github.com
  IdentityFile ~/.ssh/gupurs
  User git' >> ~/.ssh/config
4. Genel Anahtarı Görüntüleme
GitHub'a eklemek için genel anahtarınızı görüntülemek isterseniz:

cat ~/.ssh/gupurs.pub
Bu komutun çıktısını kopyalayıp GitHub hesabınızın SSH anahtarları bölümüne ekleyebilirsiniz.

5. Bağlantıyı Test Etme
GitHub bağlantınızı test etmek için:

ssh -T git@github.com
Bu adımları takip ederek, Jetson Nano'da "gupurs" adında bir SSH anahtarı oluşturup, GitHub'daki gizli repolarınıza erişim için kullanabilirsiniz.