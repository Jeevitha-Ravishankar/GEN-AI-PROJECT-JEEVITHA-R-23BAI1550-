# GEN-AI-PROJECT-JEEVITHA-R-23BAI1550-


# ImagiQ: Smart Captioning, Hashtags, Mood Detection & Visual Q&A

ImagiQ is a web-based application that uses state-of-the-art Generative AI and transformer-based models to understand and describe images intelligently. It enables users to:

- Generate image captions  
- Create relevant hashtags  
- Detect the mood/emotion of a caption  
- Ask natural language questions about the image  
- Play captions using Text-to-Speech  
- Evaluate mood detection accuracy  

## Features

- Image Captioning: Uses BLIP to generate context-aware image captions  
- Hashtag Generation: Extracts keywords from the caption to form hashtags  
- Mood Detection: Uses DistilBERT to detect emotions in captions  
- Visual Question Answering: Ask questions about the uploaded image  
- Text-to-Speech: Play the caption audio using gTTS  
- Evaluation Module: Compare true and predicted moods using accuracy, F1-score, and confusion matrix  

## Tech Stack

- Frontend/UI: Streamlit  
- Models: HuggingFace Transformers (BLIP, DistilBERT, GPT-2)  
- Other Libraries: PyTorch, gTTS, scikit-learn, PIL, matplotlib  
