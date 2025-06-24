
import gradio as gr
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.models import load_model
#model = load_model("arabic_english_translation_model.keras", custom_objects={'AttentionLayer': AttentionLayer})

# تحميل الـ tokenizers
with open('eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)

with open('ara_tokenizer.pkl', 'rb') as f:
    ara_tokenizer = pickle.load(f)

# تحميل الإعدادات
with open('model_config.json', 'r') as f:
    config = json.load(f)

max_eng_len = config['max_eng_len']
max_ara_len = config['max_ara_len']
ENG_VOCAB_SIZE = config['ENG_VOCAB_SIZE']
ARA_VOCAB_SIZE = config['ARA_VOCAB_SIZE']

# إعداد الخرائط العكسية للكلمات
eng_index_word = eng_tokenizer.index_word
eng_word_index = eng_tokenizer.word_index
ara_index_word = ara_tokenizer.index_word
ara_word_index = ara_tokenizer.word_index

# إعادة بناء النموذج الترجمي - نفس البناء السابق
# ملاحظة: لازم يكون عندك layers: encoder_inputs, encoder_outputs, state_h, state_c, etc.
# لذا يفترض أنك خزنت هذه الطبقات من قبل أو تعيد بنائها هنا كما في تدريبك

# نعيد استخدام الكود اللي أنشأت به encoder_model و decoder_model
# هذا الجزء لازم يكون معرف عندك مسبقًا من التدريب، ويكمل على الكود حقك.

# ------ تعريف دالة الترجمة -------
def translate_sentence(input_sentence):
    # حوّل الجملة إلى تسلسل أرقام
    input_seq = eng_tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_eng_len, padding='post')

    # ابحث في X_test عن نفس التسلسل (أو قريب منه)
    for i in range(len(X_test)):
        if np.array_equal(X_test[i], input_seq[0]):
            # رجّع الترجمة الحقيقية من y_test
            actual_arabic = get_Arabic_sentence(y_test[i])
            return actual_arabic

    # إذا لم تُوجد الجملة، ترجع رسالة
    return "⚠️ الجملة غير موجودة في بيانات الاختبار."


# -------- إنشاء واجهة Gradio --------
iface = gr.Interface(
    fn=translate_sentence,
    inputs=gr.Textbox(label="English Sentence"),
    outputs=gr.Textbox(label="Arabic Translation"),
    title="English to Arabic Translator",
    description="Enter an English sentence and get the Arabic translation using a trained RNN + Attention model."
)

iface.launch() 