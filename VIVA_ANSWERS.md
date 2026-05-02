Q: What is your project about?
A: My project is an AI-based anemia screening system that analyzes conjunctiva (inner eyelid) images and predicts whether a person is likely Anaemic or Non-Anaemic. It uses an EfficientNetB0 transfer learning model trained on 401 labeled eye images. The web app is built with Flask and also provides GradCAM explainability so users can see which regions influenced the prediction. It is designed as a fast, non-invasive screening aid, not a final diagnosis tool.

Q: Why did you choose anemia detection?
A: Anemia is very common and often underdiagnosed in early stages, especially where frequent lab testing is not easily accessible. A camera-based AI screening approach can help identify risk early and encourage timely medical follow-up. The conjunctiva is clinically relevant because pallor in this region is associated with low hemoglobin. So this problem has both medical relevance and practical impact.

Q: What is transfer learning and why did you use it?
A: Transfer learning means starting from a model pre-trained on a large dataset and adapting it to a specific task. I used it because my medical image dataset is relatively small (401 images), and training a deep model from scratch would overfit and perform poorly. EfficientNetB0 already learns strong visual features, so transfer learning improves convergence and generalization. It also reduces training time and compute cost.

Q: Why EfficientNetB0 specifically?
A: EfficientNetB0 gives a strong accuracy-efficiency balance with fewer parameters than many older CNNs. It is lightweight enough for practical deployment while still capturing fine visual texture and color cues needed for conjunctiva analysis. It also integrates well with TensorFlow/Keras and transfer learning workflows. For this project scale, it provided stable training and solid performance.

Q: What is the difference between your two training stages?
A: In stage one, I froze the EfficientNet base and trained only the custom classification head for 15 epochs, so the new top layers could adapt to anemia-specific features. In stage two, I unfroze the top 30 layers of the base model and fine-tuned them with a lower learning rate. This lets higher-level pretrained features adjust to the medical domain without destroying useful generic features. The two-stage process improved task-specific performance more reliably than a single-stage approach.

Q: What is class weighting and why was it needed?
A: Class weighting increases the training importance of underrepresented or harder classes during loss calculation. I used anaemic=1.42 and non-anaemic=0.90 so the model would pay more attention to the anaemic class and reduce bias toward the majority class. This is important in screening because missing true anaemic cases is a higher-risk error. It helped improve sensitivity, reflected in 75% anaemic recall.

Q: Why did you NOT divide by 255 in preprocessing?
A: I did not manually divide by 255 because EfficientNet preprocessing behavior is handled in the model pipeline used during training/inference consistency in this project setup. The input is resized to 224x224 and converted to float32 while retaining the 0-255 range to match how the model was trained. If preprocessing at inference differs from training, performance can drop significantly. So keeping the same preprocessing contract is critical for reliable predictions.

Q: What is GradCAM and how does it work in your project?
A: GradCAM is an explainable AI technique that highlights spatial regions most relevant to a model decision. In my project, I generate a heatmap from EfficientNetB0 `top_conv` activation maps and overlay it on the conjunctiva image. Red/yellow zones indicate where the network focused most strongly. This improves interpretability and helps users understand why a prediction was made.

Q: What is your model accuracy and is it good enough?
A: The model achieved 86.9% overall accuracy with 75% anaemic recall. For an image-based screening prototype, this is a strong result and shows practical usefulness. However, it is still not a replacement for clinical diagnostics, especially in high-stakes decisions. So it is good enough for preliminary risk screening, not for definitive diagnosis.

Q: What is the self-learning pipeline?
A: The self-learning pipeline automatically stores high-confidence predictions to support future model improvement. In my system, predictions are considered only when confidence is at least 92%, and retraining is triggered in batches of 20 images. This creates a controlled feedback loop rather than blindly learning from all predictions. It is designed to incrementally improve data coverage while limiting noisy pseudo-labels.

Q: What are the limitations of your project?
A: The dataset size is limited, so demographic, device, and lighting diversity may still be insufficient. Performance depends on image quality and correct conjunctiva visibility; poor angle or low light can cause uncertain outputs. It is an image-based screening model, so it cannot directly measure hemoglobin concentration like laboratory methods. Also, domain validation across broader clinical settings is still required.

Q: How is this different from a blood test?
A: A blood test directly measures biomarkers such as hemoglobin and is the diagnostic gold standard. My system only infers anemia risk from visual patterns in conjunctiva images. So this tool is non-invasive and fast, but indirect and less definitive. It should be used to assist early screening and triage, then confirmed with clinical blood tests.

Q: What is conjunctiva and why analyze it?
A: The conjunctiva is the thin mucosal tissue lining the inner eyelid and covering part of the eye surface. Clinically, conjunctival pallor is a known sign associated with anemia and reduced hemoglobin. It is relatively easy to image with a phone camera compared to invasive sampling. That makes it a practical target for computer vision-based screening.

Q: What would you improve with more time?
A: I would expand the dataset with more balanced and diverse samples across age groups, skin tones, devices, and lighting conditions. I would add stronger calibration and threshold optimization for better clinical sensitivity-specificity trade-offs. I would also integrate external validation on independent hospital datasets and improve uncertainty estimation. Finally, I would build a full MLOps retraining/monitoring pipeline for safer long-term deployment.

Q: Is your system safe to use medically?
A: It is safe only as a screening support tool with clear disclaimers, not as a standalone medical decision system. The app already labels itself as non-diagnostic and advises doctor consultation, especially for anaemic or uncertain results. This is important to prevent over-reliance and false reassurance. Clinical confirmation with blood tests remains mandatory before treatment decisions.
