from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50
from keras.optimizers import Adam

# สร้าง ImageDataGenerator สำหรับ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,               # ปรับขนาดค่าพิกเซลให้อยู่ในช่วง 0-1
    rotation_range=20,            # หมุนภาพได้ 20 องศา
    width_shift_range=0.2,        # เลื่อนภาพในแนวนอน 20%
    height_shift_range=0.2,       # เลื่อนภาพในแนวตั้ง 20%
    shear_range=0.2,              # เปลี่ยนรูปร่าง
    zoom_range=0.2,               # ย่อขยายภาพ
    horizontal_flip=True,         # พลิกภาพในแนวนอน
    fill_mode='nearest'           # เติมพื้นที่ว่างหลังจากการหมุนหรือย้าย
)

# สร้างโมเดล
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# สร้างโมเดลใหม่
model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(500, activation='softmax')  # 500 คน (500 คลาส)
])

# คอมไพล์โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# สร้าง generator สำหรับฝึก
train_generator = datagen.flow_from_directory(
    'AiCelebrity',  
    target_size=(1024, 1024),  # ปรับขนาดภาพให้เหมาะสมกับโมเดล
    batch_size=32,
    class_mode='categorical'  # ใช้สำหรับการจำแนกหลายคลาส
)

# ฝึกโมเดล
history = model.fit(
    train_generator,
    epochs=10,  # จำนวนรอบที่ฝึก
    steps_per_epoch=len(train_generator),  # จำนวนขั้นตอนในแต่ละ epoch
    verbose=1
)

# แสดงผลลัพธ์ความแม่นยำหลังการฝึก
print("Training Accuracy: ", history.history['accuracy'][-1])  # แสดงค่าความแม่นยำสุดท้าย
