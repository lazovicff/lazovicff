import React, { useEffect, useRef, useState } from "react";
import css from "styled-jsx/css";
import { loadData, createModel, trainModel, predict, loadModel } from "../utils/model";
import dynamic from "next/dynamic";
import next from "next";

const Sketch = dynamic(() => import("react-p5"), {
	ssr: false,
});

const Canvas = () => {
	const [state, setState] = useState({
		pred: null,
		examplePreds: [null, null, null, null, null],
		noisePred: [null, null, null, null, null],
	});
	const p5Cache = useRef(null);
	const imgs = useRef([]);
	const noiseImgs = useRef([]);

	const preload = (p5) => {
		const img0 = p5.loadImage("/1-final.jpg");
		const img1 = p5.loadImage("/3-final.jpg");
		const img2 = p5.loadImage("/5-final.jpg");
		const img3 = p5.loadImage("/4-final.jpg");
		const img4 = p5.loadImage("/0-final.jpg");
		imgs.current = [img0, img1, img2, img3, img4];

		const nimg0 = p5.loadImage("/1-noise-final.jpg");
		const nimg1 = p5.loadImage("/3-noise-final.jpg");
		const nimg2 = p5.loadImage("/5-noise-final.jpg");
		const nimg3 = p5.loadImage("/4-noise-final.jpg");
		const nimg4 = p5.loadImage("/0-noise-final.jpg");
		noiseImgs.current = [nimg0, nimg1, nimg2, nimg3, nimg4];
	}

	const setup = (p5, parentRef) => {
		p5.pixelDensity(1);
		p5.createCanvas(400, 400).parent(parentRef);
		p5.background(0);
		p5.fill(255);
		p5.stroke(255);

		loadModel().then(model => {
			const gsarr = imgs.current.map(img => {
				img.resize(28, 28);
				img.loadPixels();
				const grayScalePixels = getGrayScaleArray(img.pixels);
				return grayScalePixels;
			});

			const gsarr1 = noiseImgs.current.map(img => {
				img.resize(28, 28);
				img.loadPixels();
				const grayScalePixels = getGrayScaleArray(img.pixels);
				return grayScalePixels;
			});

			const result = predict(gsarr, model);
			const result1 = predict(gsarr1, model);
			return Promise.all([result.argMax(-1).array(), result1.argMax(-1).array()]);
		}).then(([exampleArr, noiseArr]) => {
			setState({
				...state,
				examplePreds: exampleArr,
				noisePred: noiseArr
			});
		});

		p5Cache.current = p5;
	};

	const draw = p5 => {
		if (p5.mouseIsPressed) {
			p5.ellipse(p5.mouseX, p5.mouseY, 40);
		}
	};

	const onClear = () => {
		if (p5Cache.current) {
			p5Cache.current.background(0);
		}
	};

	const getGrayScaleArray = (pixels) => {
		const grayScalePixels = [];
		for (let i = 0; i < pixels.length; i += 4) {
			let red = pixels[i];
			let blue = pixels[i + 1];
			let green = pixels[i + 2];
			let alpha = pixels[i + 3];
			let gray = (red + blue + green + alpha) / 4;
			grayScalePixels.push(gray / 255);
		}
		return grayScalePixels;
	}

	const onClassify = async () => {
		let img = p5Cache.current.createImage(400, 400);
		img.loadPixels();
		p5Cache.current.loadPixels();

		for (let i = 0; i < p5Cache.current.pixels.length; i += 1) {
			img.pixels[i] = p5Cache.current.pixels[i];
		}

		img.updatePixels();
		p5Cache.current.updatePixels();
		img.resize(28, 28);

		const grayScalePixels = getGrayScaleArray(img.pixels);

		loadModel().then(model => {
			const result = predict([grayScalePixels], model);
			result.print();
			return result.argMax(-1).array();
		}).then(arr => {
			setState({
				...state,
				pred: arr[0]
			});
		})
	};

	const onTrain = async () => {
		const data = await loadData();
		const model = createModel();
		const res = await trainModel(model, data);
		console.log(res.history);
		await model.save('downloads://mnist-model');
	};

	return (
		<div className="canvas">
			<h1>MNIST demo</h1>
			<p>Draw a digit in a canvas below:</p>
			<div className="sketch">
				<Sketch setup={setup} draw={draw} preload={preload} />
			</div>
			<div>
				<h2>Prediction: {state.pred}</h2>
			</div>
			<div>
				<button className="button" onClick={onClassify}>PREDICT</button>
				<button className="button" onClick={onClear}>CLEAR</button>
			</div>
			<div className="real-world-example">
				<p>Real world example:</p>
				<img className="example-digits" src="/example-digits.jpg" />
			</div>

			<div className="real-world-example-edited">
				<div className="digit">
					<img className="digit-img" src="/1.jpg" />
					<img className="digit-img" src="/1-edited.jpg" />
					<img className="digit-img" src="/1-final.jpg" />
					<p>Class: {state.examplePreds[0]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/3.jpg" />
					<img className="digit-img" src="/3-edited.jpg" />
					<img className="digit-img" src="/3-final.jpg" />
					<p>Class: {state.examplePreds[1]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/5.jpg" />
					<img className="digit-img" src="/5-edited.jpg" />
					<img className="digit-img" src="/5-final.jpg" />
					<p>Class: {state.examplePreds[2]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/4.jpg" />
					<img className="digit-img" src="/4-edited.jpg" />
					<img className="digit-img" src="/4-final.jpg" />
					<p>Class: {state.examplePreds[3]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/0.jpg" />
					<img className="digit-img" src="/0-edited.jpg" />
					<img className="digit-img" src="/0-final.jpg" />
					<p>Class: {state.examplePreds[4]}</p>
				</div>
			</div>
			<div>
				<p>Adversarial noise examples:</p>
			</div>
			<div className="real-world-example-noise">
				<div className="digit">
					<img className="digit-img" src="/1-noise-final.jpg" />
					<p>Class: {state.noisePred[0]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/3-noise-final.jpg" />
					<p>Class: {state.noisePred[1]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/5-noise-final.jpg" />
					<p>Class: {state.noisePred[2]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/4-noise-final.jpg" />
					<p>Class: {state.noisePred[3]}</p>
				</div>
				<div className="digit">
					<img className="digit-img" src="/0-noise-final.jpg" />
					<p>Class: {state.noisePred[4]}</p>
				</div>
			</div>
			<style jsx>{styles}</style>
		</div>
	);
};

const styles = css`
.canvas {
	width: 400px;
	margin: auto;
}

.sketch {
	touch-action: none;
}

.button {
	padding: 20px;
	background: #007fff;
	border: none;
	border: 1px solid #0062c4;
	color: white;
	width: 200px;
	font-size: 20px;
	box-sizing: border-box;
}

.right-button:hover {
	background: #0062c4;
	cursor: pointer;
}

.left-button:hover {
	cursor: pointer;
	background: #0062c4;
}

.left-button:focus {
	outline: none;
}

.right-button:focus {
	outline: none;
}

.example-digits {
	width: 400px;
}

.real-world-example {
	padding: 40px 0;
}

.real-world-example-edited {
	display: flex;
	justify-content: space-between;
	width: 400px;
	padding-bottom: 40px;
	margin: auto;
}

.real-world-example-noise {
	display: flex;
	justify-content: space-between;
	width: 400px;
	padding-bottom: 40px;
	margin: auto;
}

.digit {
	width: 70px;
}

.digit-img {
	width: 70px;
	height: 70px;
}
`;

export default Canvas;