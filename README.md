# clustering
Text analytics using unsupervised learning (K-means clustering)

The objective of this project is to group sentences having similar context but not exactly the same to be grouped under a common bucket.

Text processing involves removal of noise data, POS tagging, using n-grams(to maintain the context of the sentence) and text extraction using RegEx that are present in the given script.
After the data is pre-processed, we need to apply the K-means algortihm which is an unsupervised method to group all messages having similar context into a common cluster.
Before applying the algorithm, it is important to figure out the optimal number of clusters for which the algortihm needs to run on. For this purpose, we have used the elbow method by running the algo on a range of numbers and then plotting a graph of  within cluseter sum of squares(WCSS)and the number of clusters given in figure 'elbow_method.png'.
The final output gets generated in the 'clustered_final_1.csv' file.



