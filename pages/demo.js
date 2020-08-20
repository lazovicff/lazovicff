import React from "react";
import Canvas from "../components/canvas";

const App = () => (
	<div className="app">
		<Canvas />
		<style jsx global>{`
			html {
				padding: 0;
				margin: 0;
			}

			body {
				padding: 0;
				margin: 0;
			}

			#__next {
				padding: 0;
				margin: 0;
			}

			.app {
				text-align: center;
			}
		`}</style>
	</div>
);

export default App;