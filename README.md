# Diseased Leaves Recognition

Nhận biết lá cây tại trường sư phạm TP.Hồ Chí Minh có bị bệnh hay không bệnh. 

- Trạng thái không bệnh là những loại lá mang màu sắc gốc của nó theo đúng giai đoạn phát triển (không tính giai đoạn lá héo). Chúng nguyên vẹn, không rách hay bị sâu bệnh.
- Trạng thái không bình thường: là những chiếc lá không còn nguyên vẹn, bị sâu bệnh, bị héo.

## Dataset:
Data sẽ được thu thập từ 2 nguồn:
- Dataset được thu thập bằng tay ở trường đại học Sư Phạm HCM ([link download](https://drive.google.com/file/d/1xzOQ3GN9mRheYqG6RLg4a93tgGl6CxcM/view?usp=sharing)). Data được thu thập và gán nhãn bằng tay từ 16 loại cây.

- Dataset hỗ trợ: Cuộc thi Plant Pathology 2020 - FGVC7 trên Kaggle ([link cuộc thi](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/)).

## Dependencies
- numpy==1.19.4
- pandas==1.2.0
- matplotlib==3.4.2
- opencv-python==4.4.0.42
- scikit-learn==0.24.0
- tensorflow==2.5.0
- notebook==6.4.0
- image-classifiers==1.0.0
- tqdm==4.61.0

## Use code
1. **Prepare dataset:**
    ```
    python "drive/MyDrive/Colab Notebooks/[Vision]Diseased leaves recognition/src/prepare_dataset.py" \
    --raw_data 'diseased-leaves-recognition-dataset/' \
    --extrernal_data 'plant-pathology-2020-fgvc7/' \
    --processed_data ''
    ```

2. **Training:**
   ```
    python "drive/MyDrive/Colab Notebooks/[Vision]Diseased leaves recognition/src/training.py" \
    --processed_data '' \
    --epochs 50 \
    --save_model ''
   ```

---