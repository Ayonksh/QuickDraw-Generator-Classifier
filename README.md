# QuickDraw-Generator-Classifier

使用 CDCGAN 生成图片，使用 ResNet34 对图片进行分类。数据集来自 [Quick, Draw](https://github.com/googlecreativelab/quickdraw-dataset)

## 效果

|               airplane               |              bicycle               |               butterfly                |             cake             |              camera              |
| :----------------------------------: | :--------------------------------: | :------------------------------------: | :--------------------------: | :------------------------------: |
| ![airplane](assets/gif/airplane.gif) | ![bicycle](assets/gif/bicycle.gif) | ![butterfly](assets/gif/butterfly.gif) | ![cake](assets/gif/cake.gif) | ![camera](assets/gif/camera.gif) |

|             chair              |             clock              |              diamond               |                 The Effiel Tower                 |             tree             |
| :----------------------------: | :----------------------------: | :--------------------------------: | :----------------------------------------------: | :--------------------------: |
| ![chair](assets/gif/chair.gif) | ![clock](assets/gif/clock.gif) | ![diamond](assets/gif/diamond.gif) | ![TheEffielTower](assets/gif/TheEffielTower.gif) | ![tree](assets/gif/tree.gif) |

## 生成器

Model: CDCGAN

|          |                    Discriminator Loss                    |                    Generator Loss                    |                 Result                  |
| :------: | :------------------------------------------------------: | :--------------------------------------------------: | :-------------------------------------: |
| airplane | ![Discriminator Loss](assets/D_Loss_CDCGAN_airplane.png) | ![Generator Loss](assets/G_Loss_CDCGAN_airplane.png) | ![airplane](assets/airplane_CDCGAN.png) |
|  camera  |  ![Discriminator Loss](assets/D_Loss_CDCGAN_camera.png)  |  ![Generator Loss](assets/G_Loss_CDCGAN_camera.png)  |   ![camera](assets/camera_CDCGAN.png)   |

## 分类器

Model: ResNet34  

|          |                        Train                        |                       Test                        |
| :------: | :-------------------------------------------------: | :-----------------------------------------------: |
|   Loss   |   ![Train Loss](assets/classifier_train_loss.png)   |   ![Test Loss](assets/classifier_test_loss.png)   |
| Accuracy | ![Train Accu](assets/classifier_train_accu.png) 99% | ![Test Accu](assets/classifier_test_accu.png) 96% |

## 使用方法

+ 下载数据和生成数据集

  ```bash
  cd DataUtils
  python dataset_loader.py
  ```

+ 开始训练

  1. CDCGAN

     ```bash
     cd Generation
     python cdcgan.py
     ```

  2. ResNet34

     ```bash
     cd Classification
     python resnet34.py
     ```

+ 图片分类

  ```
  python quickdraw.py
  ```

