import numpy as np
from keras.models import load_model
from keras.models import Model

filepath = "weightsBest.hdf5"
unknownVector = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)

# 4. 模型的加载及使用其中某一层作为输出
print("Using loaded model to predict...")
model = load_model(filepath)
model.summary()
for i in model.layers:
    print(i.name)
# 一共有5层，使用第3层作为输出
intermediateLayey = Model(inputs=model.input,
                          outputs=model.get_layer('dense_3').output)  # 创建的新模型
# 一共有5层，使用最后一层作为输出
ouputLayer = Model(inputs=model.input,
                   outputs=model.output)  # 创建的新模型

intermediateLayeyPredicted = intermediateLayey.predict(unknownVector)
print("使用第3层作为输出,第三层的输出向量为{vector}，输出数据的长度为{length}"\
      .format(vector=str(model.get_layer('dense_3').output_shape),length=intermediateLayeyPredicted.size))
print("\n第3层的数据输出")
print(intermediateLayeyPredicted)
ouputLayerPredicted = ouputLayer.predict(unknownVector)
print("\n使用最后一层作为输出")
print(ouputLayerPredicted)
print("\n输出值最大的位置是:({})".format(np.argmax(ouputLayerPredicted)+1))


