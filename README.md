# Embracing the Modern Era of Iris Quality Assessment utilizing Feature Space

![Logo](https://github.com/user-attachments/assets/1d0477fc-8df2-4790-be86-c284f8b996a4)

***


## Introduction

In an era where biometric security plays a vital role in safeguarding personal data and ensuring secure access, the quality of iris images has become a cornerstone for accurate and reliable identification.

This project represents our team’s endeavor to develop a model capable of assessing iris image quality. We designed a machine learning pipeline that emphasizes precision and robustness. Our system employs Python as the primary programming language, incorporating cutting-edge libraries such as Torch, OpenCV to preprocess, build, and evaluate the quality assessment model.

The application is lightweight, scalable, and tailored for practical deployment in various biometric systems, such as secure access control, e-passport verification, and mobile authentication. It focuses on balancing high performance with computational efficiency, making it suitable for both on-device and cloud-based implementations.

***This project is developed and redesigned from scratch based on the project [Recognition Oriented Iris Image Quality Assessment in the Feature Space](https://arxiv.org/abs/2009.00294) belongs to Leyuan Wang and her colleagues. Hence, it is intended for `educational purposes only`. The model's performance is influenced by dataset constraints and may not guarantee absolute accuracy in all scenarios. It is not designed to replace professional-grade biometric systems or compliance frameworks.***

***


## Key features

### Hand-crafted Based Method  
This approach uses predefined criteria to evaluate iris image quality. Each image is scored based on specific metrics and filtered using fixed thresholds.  

- **Criteria Determining:** Criteria derived from prior studies are processed and analyzed using correlation plots and feature selection via Random Forest to identify the most relevant metrics.  
- **Threshold Assesing:** Thresholds for each criterion are manually analyzed to establish the minimum quality requirements.  
- **Relations plotted:** SHAP (SHapley Additive exPlanations) and PDP (Partial Dependence Plots) are employed to uncover nonlinear relationships among the selected criteria.  
- **Fusion Function Creating:** A fusion function is developed, leveraging the discovered nonlinear relationships to aggregate the selected criteria effectively.  

---

### DFS-based Method  
This approach employs deep learning to assess iris image quality by measuring the distance in the feature space (Distance in Feature Space - DFS).  

- **Feature Extraction:** MobileNetV2, a lightweight and efficient model, is utilized to extract high-level features from iris images.  
- **Semantic Segmentation:** U-Net and LR-ASPP are integrated to perform segmentation and focus specifically on the iris region.  
- **Attention Mechanism:** Heatmaps are used to emphasize critical regions, minimizing interference from surrounding noise such as eyelashes.  
- **Quality Scoring:** Weighted pooling is applied to combine information from the feature map and compute the final quality score.  

***


## Project Structure

- **dfs/**
  - **\_\_pycache\_\_/**: Python cache files.
  - **test_image_casia/**: Some images used for testing
  - **weights/**:
    - `1211_202056_MobileNetV2_Lite...`: Pretrained DFS model utlizing MobileV2.
    - `iris_semseg_upp_scse_mobilen...`: Pretrained model use for segmentaion.
  - `attention.py`: Source code for the Attention mechanism.
  - `decoder.py`: Source code for the decoder.
  - `encoder.py`: Source code for the encoder.
  - `metric.py`: Source code for calculating evaluation metrics relating to the hand - craft process.
  - `quality_score.py`: Source code for calculating quality scores of DFS.
- **hand_craft/**
  - **criteria_choosing/**: Folder containing criteria choosing pipeline.
  - **criteria_final/**: Folder containing criteria concatenating pipeline.
- `infer.py`: Source code for performing inference.

## Notes
- The `test_image_casia` directory contains sample images for testing the model.
- The `weights` directory stores pre-trained model weights.
- If you only require the model's output, you only need to work with the infer.py file, where running the script will automatically launch the LocalHost interface.
  
***


## Usage

Please refer to these following links for essential document:

[Dataset Link](https://drive.google.com/file/d/1KCMY3_eloUE7_BKlmzA2bfq5aryXoIbv/view?usp=sharing)

Once you have successfully cloned this repository, to use our interface, please run the `infer.py` file and follow the interface instructions provided.

This is our original interface, where you will need to upload the image you want to evaluate.
![original-interface1](https://github.com/user-attachments/assets/db5473f7-5916-42da-a57c-af1cc48327e9)
![original-interface2](https://github.com/user-attachments/assets/a836b0d2-7e26-45bc-aedb-58f320a47bf8)

1. Upload the image you wish to assess into the interface.
2. Choose one of the two available methods:
   - **Hand-Crafted Based Method**: The threshold scores for each criterion are predefined in the interface. The fusion score threshold is set to **0.6**.
![hc-interface1](https://github.com/user-attachments/assets/afdc0b59-08bd-4c9b-be9c-3cde9c636163)
![hc-interface2](https://github.com/user-attachments/assets/7c40dab3-31ac-4812-9e3e-35d5b7e05d07)

   - **DFS-Net Based Method**: The evaluation threshold is set to **0.9**.
![dfs-interface1](https://github.com/user-attachments/assets/5a8a7351-9ed7-4760-bf07-e1171a0883e4)
![dfs-interface2](https://github.com/user-attachments/assets/99cc0655-a0da-4756-9394-f45972be35fd)

***


## Acknowledgments
This project would not have been possible without the invaluable contributions of several open-source libraries, such as [Keras](link Keras), [PyTorch](link Pytorch), and [Diffusers](link Diffusers). Their robust tools and resources were instrumental in the success of this project.

We extend our heartfelt gratitude to our lecturers, Ph.D. Pham Ngoc Hung, for assigning us this challenging yet captivating project. It has been an incredible learning opportunity that has significantly enhanced our knowledge and skillset.

Finally, we would like to acknowledge AI communities in general for their indirect contributions, offering both moral and practical support that kept us motivated, espically Leyuan Wang and her colleagues colleagues in the project [Recognition Oriented Iris Image Quality Assessment in the Feature Space](https://arxiv.org/abs/2009.00294) which we had the opportunity to study and replicate.
.

***


## Contributors
- Vũ Như Đức - 20225485
- Lại Trí Dũng - 20225486
- Vũ Quốc Dũng - 20225488

***


## License
This project is licensed under the [MIT License](LICENSE).
