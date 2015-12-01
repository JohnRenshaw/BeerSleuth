# BeerSleuth

Beer Sleuth is a recommendation engine which can predict how you would rate the taste of beer based on ratings you have given other beers. It works using a method called collaboritive filtering to train the model and generate predictions. Specifically the recommender is based on non-negative matrix factorization, solved with alternating least squares. I used Apache Spark to build the model and the project is being hosted on Amazon Web Services. The training data for the model was scraped from ratebeer.com which has millions of reviews of differnt beers and a lot of really great information about the beers such as style, brewery or alcohol by volume. I scraped over 1.8 million reviews from the site and should have every ratings for over 60,000 beers. I should have every beer from every brewery CA, WA, OR, CO, MA, GA, AL anf FL. The root mean squared error of the test data is about 1.08, put simply, on a scale of 1-10 Beer Sleuth is about 1 away from how you would rate the taste of the beer on average.

Beer Sleuth was created as my capstone project for the Galvanize data data science immersive program, which I would highly reccomend for anyone trying to transition into a career in data science. I hope all of you find this project work helpful or at least amusing.

This project is hosted at beersleuth.net
