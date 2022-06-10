# Detectron2를 활용한 Object detection 기술 연구

# Professor
김휘용

# Student
2015104181 서승우

# 목표
Detectron2를 활용하여 Mask R-CNN 모델 학습을 통해 자동차 번호판을 검출

# Process
1. Find Dataset(car license plate)
2. Create json file 
3. Train using google_colab
4. Check loss function graph
5. Early stopping if overfitting occurs
6. Check Mean Average Precision 
7. Test 


# Dataset
![Dataset_image](https://user-images.githubusercontent.com/101958056/173061521-c9a2cb0e-6e21-4657-87ba-45010a616763.png)

# Labeme(for json_file)
![Labelme_image](https://user-images.githubusercontent.com/101958056/173061681-629ed25b-1000-4039-ad58-bf3b2f7ed2bc.png)

# Car_license_plate_sample_image_1
![sample image_1](https://user-images.githubusercontent.com/101958056/172886846-42c09c77-eee7-4ecd-9a6d-d201e1559d3b.png)

# Car_license_plate_sample_image_2
![sample image_2](https://user-images.githubusercontent.com/101958056/172887558-14ffa91c-cf5a-40de-a7df-abffa9356089.png)


# 학습 

cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.00025    

cfg.SOLVER.MAX_ITER = 5000 

# Train loss function (5k)
![train loss function 5k](https://user-images.githubusercontent.com/101958056/172887623-365aa1fa-dd77-4281-bc16-24e740a5a4e7.png)

# Validation loss fucntion (5k)
![validation loss fucntion 5k ](https://user-images.githubusercontent.com/101958056/172887632-8fbca997-c9d1-4670-9366-d7f49542f05e.png)

# Mean Average Precision (5k)
![mAP](https://user-images.githubusercontent.com/101958056/172887639-98c809d7-115d-4320-951d-3c9d45a893e4.png)

# Test picture (5k)
![5k test picture](https://user-images.githubusercontent.com/101958056/172887636-078382bd-ea81-46a1-9e74-f35956f4bb77.png)

# Tool
Labelme 

Google Colab

pyyaml

pytorch

cuda

torchvision

detectron2


