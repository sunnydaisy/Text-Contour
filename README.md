Text-Contour
===============
* 머신러닝과 OpenCV 활용한 글자 검출
  * 외국인을 위한 메뉴판 번역 기능에 추가하는 용도이므로 메뉴판 글씨 인식에 초점을 맞춤  
  * 이미지 내에서 글자 윤곽선을 찾아 그룹으로 추출해 내기 (텍스트 덩어리 추출)  
* naver d2 참조 https://d2.naver.com/helloworld/8344782

이미지 수집
----------
* 구글 이미지 수집 라이브러리인 google_images_download 사용
* 이미지 수집 대상
  > 창문, 창틀, 모델하우스, 자막, 포스터, 메뉴판, 맛집, homebaking, 메뉴판 디자인, 메뉴판 일러스트, 음식 메뉴 아이콘
  * 창문 틀, 상자, 안경과 같은 이미지도 바운딩 박스처리 되므로 제거 필요
  * 글자 이미지는 1폴더 글자가 아닌 이미지는 0폴더에 저장

영상처리
-----------
1.gray 변환
  * 글자 검출에는 색이 필요 없으므로 회색조(gray scale) or 흑백(binary)로 변환하여 추출
  
2.모폴로지기법 MORPH_GRADIENT
  * dilation과 erosion 이미지 차이
  
3.Adaptive threshold  
  * 임계값 적용하면 불필요한 영역 잡영(noise) 제거하는 효과 있어 사물 탐지할 때 Contour 추출 가능  
    * original, global, mean, gaussian 중 mean 선택  
    
4.morphology close  
  * Opening과 반대로 Dilation 연산을 먼저 적용한 후,  Erosion 연산을 적용한다.  
  * 오브젝트에 있는 작은 검은색 구멍들을 메우는데 사용  
  
5.houghlinesp  
  * 허프변환 통해 직선 찾기  
  
6.bounding box 처리  
  * 바운딩 박스부분만 따로 이미지 저장  
  * 너무 작은 이미지는 수집되지 않도록 함  

* NMS(non_max_suppression) 함수 사용

<div>
 <img width="320" alt="_11 21preprocessing9" src="https://user-images.githubusercontent.com/47199328/91385940-e851ac00-e86c-11ea-9513-28b29bd7261b.png">
 <img width="320" alt="_11 21preprocessing12" src="https://user-images.githubusercontent.com/47199328/91385945-e982d900-e86c-11ea-9357-d9f66fa0aac2.png">
 <img width="310" alt="11 22preprocessing2" src="https://user-images.githubusercontent.com/47199328/91387025-3f588080-e86f-11ea-96ca-a9c20569ee8a.png">
</div>

모델링
-----------



<img width="300" alt="img3_k10" src="https://user-images.githubusercontent.com/47199328/91385960-ee478d00-e86c-11ea-80a7-b7e9de35e8e0.png">
