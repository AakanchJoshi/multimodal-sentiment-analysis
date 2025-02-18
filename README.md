# LLMs for multimodal sentiment analysis for targeted advertising

## 🚀 Overview  
We utilized the "Multimodal EmotionLines Dataset" (MELD) to achieve our research objectives. This dataset categorizes sentiment into three classes: positive, negative, and neutral, and includes seven emotion detection classes: sad, surprise, disgust, angry, neutral,  joy, and fear. MELD enhances the original EmotionLines dataset by incorporating audio and visual modalities alongside text. Starting with the text modality, we used a speech-to-text API to extract text from videos. For the audio modality, we calculated pitch and energy from the audio files. For the visual modality, we employed Mediapipe to extract features, which were then trained using an LSTM network. The outputs from all modalities were combined using a late fusion technique, resulting in a "Prompt Template." This template was used to train an LLM via few-shot learning, enabling it to predict sentiment classes. Our research demonstrates 
that multimodal analysis significantly outperforms unimodal sentiment analysis.

## 🚀 Future Research Directions   
We determined that this research topic is well-suited for targeted advertising, but beyond this application, the product can be utilized in the following industries:

• Social Media Monitoring: By examining text, images, and videos shared on social media platforms, businesses can achieve a more profound understanding of public sentiment and opinions regarding their brands, products, or services. This analysis can 
significantly aid in reputation management, shaping marketing strategies, and  enhancing customer engagement. 
• Customer Service: In customer service settings, multimodal sentiment analysis can interpret customer emotions by analyzing their spoken words, tone of voice, and facial expressions during interactions. This enables the provision of higher quality service by 
adapting responses to align with the customer's emotional state. 
• Healthcare: In telemedicine and mental health care, multimodal sentiment analysis can assist healthcare professionals in assessing patient’s emotional states through their speech, facial expressions, and written communications. This can enhance diagnosis and treatment plans, particularly in mental health therapy.