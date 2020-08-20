import { layers, sequential, loadLayersModel, train, tidy, tensor2d } from '@tensorflow/tfjs';

import { MnistData } from './data';

let modelCache = null;

export async function loadData() {
	const data = new MnistData();
	await data.load();
	return data;
}

export function createModel() {
	const model = sequential();
	model.add(
		layers.conv2d({
			inputShape: [28, 28, 1],
			kernelSize: 5,
			filters: 32,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'VarianceScaling',
		})
	);

	model.add(
		layers.maxPool2d({
			poolSize: [2, 2],
			strides: [2, 2],
		})
	);

	model.add(
		layers.conv2d({
			kernelSize: 5,
			filters: 32,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'VarianceScaling',
		})
	);

	model.add(
		layers.maxPool2d({
			poolSize: [2, 2],
			strides: [2, 2],
		})
	);

	model.add(layers.flatten());

	model.add(
		layers.dense({
			units: 10,
			kernelInitializer: 'VarianceScaling',
			activation: 'softmax',
		})
	);

	model.compile({
		optimizer: train.sgd(0.15),
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});

	return model;
}

export function trainModel(model, data) {
	const BATCH_SIZE = 64;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
	});

	const [testXs, testYs] = tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
	});

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 10,
		shuffle: true,
	});
}

export function predict(arr, model) {
	const xs = tensor2d(arr, [arr.length, 784]);
	const reshaped_xs = xs.reshape([arr.length, 28, 28, 1])
	const preds = model.predict(reshaped_xs);

	xs.dispose();
	return preds;
}

export async function loadModel() {
	if (modelCache) {
		return modelCache;
	}
	modelCache = await loadLayersModel("https://filip.im/mnist-model.json");
	return modelCache;
}