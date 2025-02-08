function processImageData(pixels, width, height) {
    const integralMatrix = Array(height + 1).fill(0).map(() => Array(width + 1).fill(0));
    const squaredIntegralMatrix = Array(height + 1).fill(0).map(() => Array(width + 1).fill(0));

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const grayValue = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
            const normalizedValue = grayValue / 255;

            const above = integralMatrix[y][x + 1];
            const left = integralMatrix[y + 1][x];
            const aboveLeft = integralMatrix[y][x];

            const aboveSq = squaredIntegralMatrix[y][x + 1];
            const leftSq = squaredIntegralMatrix[y + 1][x];
            const aboveLeftSq = squaredIntegralMatrix[y][x];

            integralMatrix[y + 1][x + 1] = normalizedValue + above + left - aboveLeft;
            squaredIntegralMatrix[y + 1][x + 1] = (normalizedValue * normalizedValue) + aboveSq + leftSq - aboveLeftSq;
        }
    }

    return {
        integralMatrix,
        squaredIntegralMatrix
    };
}

function findFaces(integralImage, squaredIntegralImage, width, height, stepSize = 2, maxScale = 3, scaleStep = 1) {
    let rawDetections = [];

    const detectFace = (integralImage, squaredIntegralImage, posX, posY, scaleFactor) => {
        const steps = [1, 10, 100, 500, 2000, 4000, 6000]; // Cascading stages

        let sumAlphaH = 0;
        let stageIndex = 0;

        for (let i = 0; i < 6000; i++) {
            let stump = stumps[i];
            let response = stump.feature.applyFeature(integralImage, squaredIntegralImage, posX, posY, scaleFactor);

            let scaledThreshold = stump.threshold * scaleFactor * scaleFactor;
            let h = (stump.polarity * response <= stump.polarity * scaledThreshold) ? 1 : -1;
            sumAlphaH += stump.amountOfSay * h;

            if (i + 1 == steps[stageIndex]) {
                if (sumAlphaH <= 0) {
                    //console.log("Failed at stage: " + (i + 1));
                    return;
                }
                stageIndex++;
            }
        }

        rawDetections.push({
            x: posX,
            y: posY,
            scaleFactor: scaleFactor,
            confidency: sumAlphaH
        });
    }

    let scaleFactor = 1;
    while (scaleFactor <= maxScale) {
        for (let startY = 0; startY < height - ~~(WINDOW_SIZE * scaleFactor); startY += ~~(stepSize * scaleFactor)) {
            for (let startX = 0; startX < width - ~~(WINDOW_SIZE * scaleFactor); startX += ~~(stepSize * scaleFactor)) {
                detectFace(integralImage, squaredIntegralImage, startX, startY, scaleFactor);
            }
        }

        scaleFactor += scaleStep;
    }
    return filterStrongestFaces(rawDetections, WINDOW_SIZE * 0.75);
}

function filterStrongestFaces(detections, windowSize = WINDOW_SIZE / 2) {
    const filtered = [];

    detections.forEach(face => {
        let isMerged = false;

        for (let i = 0; i < filtered.length; i++) {
            let existingFace = filtered[i];
            if (
                Math.abs(face.x - existingFace.x) < windowSize &&
                Math.abs(face.y - existingFace.y) < windowSize
            ) {
                if (face.confidency > existingFace.confidency) {
                    filtered[i] = {
                        ...face
                    };
                }
                isMerged = true;
                break;
            }
        }

        if (!isMerged) {
            filtered.push({
                ...face
            });
        }
    });

    return filtered;
}