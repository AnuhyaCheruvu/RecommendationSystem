
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.*;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;

public class RecommendationEngine {

    public static void main(String[] args) {

        // Turn off unnecessary logging
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        // Create Java spark context
        SparkConf conf = new SparkConf().setAppName("Collaborative Filtering Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Read user-item rating file. format - userId,itemId,rating
        JavaRDD<String> transactionsFile = sc.textFile(args[0]);

        JavaRDD<Transaction> transactions = transactionsFile.map(new Function<String, Transaction>() {

            public Transaction call(String s) throws Exception {
                String[] sarray = s.split(",");
                String[] dateStrings = sarray[6].split(" ");
                Date date1=new SimpleDateFormat("MM/dd/yyyy").parse(dateStrings[0]);
                Transaction transaction = new Transaction(Long.parseLong(sarray[0]), sarray[1], sarray[2], sarray[3], Double.parseDouble(sarray[4]), sarray[5], date1, Long.parseLong(sarray[7]), sarray[8]);
                return transaction;
            }
        });

        List<Transaction> transactionList = transactions.collect();
        Map<Long, Map<Long, Double>> userMap = new HashMap();

        for(Transaction transaction : transactionList) {
            Long custId = transaction.getCustId();
            if (userMap.get(custId) == null) {
                Map<Long, Double> vendorMap = new HashMap<Long, Double>();
                vendorMap.put(transaction.getProductId(), 1.0d);
                userMap.put(custId, vendorMap);
            }
            else {
                if (userMap.get(custId).get(transaction.getProductId()) == null) {
                    Double rating = 1.0d;
                    userMap.get(custId).put(transaction.getProductId(), rating);
                }
                else {
                    Double current = userMap.get(custId).get(transaction.getProductId());
                    Double rating = current + (1.0d);
                    userMap.get(custId).put(transaction.getProductId(), rating);
                }
            }
        }

        try {
            PrintWriter pw = new PrintWriter(new File("pref.csv"));
            Iterator it = userMap.keySet().iterator();
            for (Map.Entry<Long, Map<Long, Double>> entry : userMap.entrySet())
            {
                Long custId = entry.getKey();
                for (Map.Entry<Long, Double> entry1 : entry.getValue().entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(custId);
                    sb.append(',');
                    sb.append(entry1.getKey());
                    sb.append(',');
                    Double rating = (entry1.getValue()/userMap.get(custId).size()) * 5.0d;
                    sb.append(rating);
                    pw.println(sb.toString());
                }
            }
            pw.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        JavaRDD<String> prefFile = sc.textFile("pref.csv");

        // Map file to Ratings(user,item,rating) tuples
        JavaRDD<Rating> ratings = prefFile.map(new Function<String, Rating>() {
            public Rating call(String s) {
                String[] sarray = s.split(",");
                return new Rating(Integer.parseInt(sarray[0]), Integer
                        .parseInt(sarray[1]), Double.parseDouble(sarray[2]));
            }
        });

//        // Build the recommendation model using ALS
//
        int rank = 10; // 10 latent factors
        int numIterations = Integer.parseInt(args[1]); // number of iterations



        MatrixFactorizationModel model = ALS.trainImplicit(JavaRDD.toRDD(ratings),
                rank, numIterations);
        //ALS.trainImplicit(arg0, arg1, arg2)
//
//        // Create user-item tuples from ratings
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings
                .map(new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                });
//
        JavaRDD<Rating> recomondations = model.predict(userProducts.rdd()).toJavaRDD().distinct();
//
//        // Sort the recommendations by rating in descending order
        recomondations = recomondations.sortBy(new Function<Rating,Double>(){
            public Double call(Rating v1) throws Exception {
                return v1.rating();
            }

        }, false, 1);

        final Long reqCustId = Long.parseLong(args[2]);

        // Get top 10 recommendations
        //JavaRDD<Rating> topRecomondations = sc.parallelize(recomondations.take(1));
        System.out.println("Top recommendations for user " + reqCustId + " :");
        recomondations.foreach(new VoidFunction<Rating>() {
            public void call(Rating rating) throws Exception {
                if (rating.user() == reqCustId) {
                    System.out.println(rating.product());
                }
            }
        });

        try {
            PrintWriter pw = new PrintWriter(new File("reco.csv"));
            List<Rating> reco = recomondations.collect();
            for (Rating rating : reco) {
                StringBuilder sb = new StringBuilder();
                sb.append(rating.user());
                sb.append(',');
                sb.append(rating.product());
                sb.append(',');
                sb.append(rating.rating());
                pw.println(sb.toString());
            }
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.print("DONE!!!!");

    }

}