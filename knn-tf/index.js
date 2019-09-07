require('@tensorflow/tfjs-node');   //running tfjs on laptop cpu
const tf = require('@tensorflow/tfjs'); //require in Tensorflow JS lib
const loadCSV = require('./load-csv');  //require file to load csv data

function knn(features, labels, predictionPoint, k) {
    const {mean, variance} = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))    //Scaled Prediction const for standardization

return (
   features
/* Step 0 - Standarization of features for  */
    .sub(mean)
    .div(variance.pow(0.5))
/* Step 1 - Find Distance between features & prediction point features */
    .sub(scaledPrediction)	//Broadcasting operation
    .pow(2)	//Elementwise operation to square each element
    .sum(1) //Sums along x(1)-axis
    .sqrt()	//Elementwise operation to take power of .5 = sqrt of each element
/* Step 2 - Sort from lowest distance to greatest distance */
    .expandDims(1) //We expand dimensions of distances tensor across the x-axis to get the shape of [4,1], the same as the labels distance
    .concat(labels, 1) //We concatenate labels to distances across the x-axis so they are linked by the same indices in one tensor
    .unstack() //We unstack our 1 tensor into 1 Vanilla JS array containing multiple tensors
/* After unstacking our tensor, we are dealing with vanilla JS array and we can ONLY use vanilla JS methods from this point on */
    .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1) //Sorting function tp sprt tensors in order of least to greatest distance
    /* Step 3 - Average the label value of the top k records */ 
    .slice(0, k)//Get Top k records
    .reduce((acc, pair) => acc + pair.get(1), 0) / k //Get average label value
)}
//Call load CSVfunction with CSV file passed in and run some preprocessing on dataset
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {  
    shuffle: true, //shuffle rows of data in CSV file to randomize test dataset
    splitTest: 10, //split test data in 2 datasets (10 for Testing/other 20,000+ for Training)
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],   //Features - Latitude/Longitude/SqFt Lot/SqFt_Living
    labelColumns: ['price']       //Label - Price of House (in Thousands of $s)
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('KNN Housing Price Prediction: $', result);     //Logging out Housing Predictions
    console.log();  //New Line
    console.log('KNN Prediction Error Percentage: %', err * 100);    //Logging out Error Percentages    
    console.log();  //New Line
})
})
