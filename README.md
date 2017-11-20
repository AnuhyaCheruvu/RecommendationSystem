# Recommendationsystem

## PRE REQUISITES
> apache-spark  version 2 above
> scala version 2 above

## HOW TO RUN
> 1. Go to the project directoy and do mvn clean install . This will genarte a jar
> 2. Start the spark context run spark-shell
> 3. execute spark-submit --class RecommendationEngine --master local[4] /Users/anuhyacheruvu/Downloads/sparktest/target/spark-test-1.0-SNAPSHOT.jar /Users/anuhyacheruvu/Downloads/testData.csv 10
>4.  this will give you an output in reco.csv which contains the 10 recommendations in descending order of rating. The reco file order is customeId, productId, rating
> 5. The input file order is same as the fiels in Transaction.java
