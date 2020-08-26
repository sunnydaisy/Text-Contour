from google_images_download import google_images_download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def imageCrawling(keyword,dir):
    response = google_images_download.googleimagesdownload()
    arguments={"keywords":keyword,"limit":100,"print_urls":True,"no_directory":True,"output_directory":dir}
    paths=response.download(arguments)

image_Path = 'F:/semigradpro/pic_pre/'
word_list = ['창문', '창틀', '모델하우스', '예능 자막', '넷플릭스 자막', '넷플릭스 영화 자막', '영화 포스터', '메뉴판', '음식판', '인스타 맛집', 'menu', 'homebaking', '메뉴판 디자인', '메뉴판 일러스트', '메뉴판 배경', '메뉴 포스터', '분식 메뉴판', '음식 메뉴 아이콘', '메뉴 일러스트']
for word in word_list:
    try:
        imageCrawling(word,image_Path)
    except:
        print('download failed')
print('Crawling Done')