
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.Boolean;
import java.lang.Double;
import java.lang.Long;
import java.sql.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.Date;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;


public class RecommendationEngine {

    private static final String s = "10";
    private static final String pattern = "MM/dd/yyyy";
    private static final String DELIMETER = ",";

    private static final String url = "jdbc:mysql://offer-recommendation-system.cko6hrw9syfz.us-west-2.rds.amazonaws.com:3306/offers";
    private static final String user = "user";
    private static final String password = "12345678";
    private static final String sqlClassName = "com.mysql.jdbc.Driver";


    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        // Create Java spark context
        SparkConf conf = new SparkConf().setAppName("Collaborative Filtering Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Read user-item rating file. format - userId,itemId,rating
        JavaRDD<String> transactionsFile = sc.textFile(args[0]);

        JavaRDD<String> itemDescritpionFile = sc.textFile(args[1]);

        JavaRDD<Transaction> transactions = transactionsFile.map(new Function<String, Transaction>() {

            public Transaction call(String s) throws Exception {
                String[] sarray = s.split(DELIMETER);
                String[] dateStrings = sarray[6].split(" ");
                Date date1 = new SimpleDateFormat(pattern).parse(dateStrings[0]);
                Transaction transaction = new Transaction(Long.parseLong(sarray[0]), sarray[1], sarray[2], sarray[3], Double.parseDouble(sarray[4]), sarray[5], date1, Long.parseLong(sarray[7]), sarray[8]);
                return transaction;
            }
        });

        List<Transaction> transactionList = transactions.collect();
        Map<Long, Map<Long, Double>> userMap = new HashMap();

        for (Transaction transaction : transactionList) {
            Long custId = transaction.getCustId();
            if (userMap.get(custId) == null) {
                Map<Long, Double> vendorMap = new HashMap<Long, Double>();
                vendorMap.put(transaction.getProductId(), 1.0d);
                userMap.put(custId, vendorMap);
            } else {
                if (userMap.get(custId).get(transaction.getProductId()) == null) {
                    Double rating = 1.0d;
                    userMap.get(custId).put(transaction.getProductId(), rating);
                } else {
                    Double current = userMap.get(custId).get(transaction.getProductId());
                    Double rating = current + (1.0d);
                    userMap.get(custId).put(transaction.getProductId(), rating);
                }
            }
        }

        try {
            PrintWriter pw = new PrintWriter(new File("pref.csv"));
            for (Map.Entry<Long, Map<Long, Double>> entry : userMap.entrySet()) {
                Long custId = entry.getKey();
                for (Map.Entry<Long, Double> entry1 : entry.getValue().entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(custId);
                    sb.append(',');
                    sb.append(entry1.getKey());
                    sb.append(',');
                    Double rating = (entry1.getValue() / userMap.get(custId).size());
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


        // Create tuples(itemId,ItemDescription), will be used later to get names of item from itemId
        JavaPairRDD<Integer, String> itemDescritpion = itemDescritpionFile.mapToPair(
                new PairFunction<String, Integer, String>() {
                    public Tuple2<Integer, String> call(String t) throws Exception {
                        String[] s = t.split(",");
                        return new Tuple2<Integer, String>(Integer.parseInt(s[0]), s[1]);
                    }
                });

//        // Build the recommendation model using ALS
//
        int rank = 10; // 10 latent factors
        int numIterations = Integer.parseInt(s); // number of iterations


        MatrixFactorizationModel model = ALS.trainImplicit(JavaRDD.toRDD(ratings),
                rank, numIterations);
//

//        // Create user-item tuples from ratings
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings
                .map(new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                });

        JavaRDD<Integer> notRatedByUser = userProducts.filter(new Function<Tuple2<Object, Object>, Boolean>() {

            public Boolean call(Tuple2<Object, Object> v1) throws Exception {
                if (((Integer) v1._1).intValue() != 0) {
                    return Boolean.TRUE;
                }
                return false;
            }
        }).map(new Function<Tuple2<Object, Object>, Integer>() {
            public Integer call(Tuple2<Object, Object> v1) throws Exception {
                return (Integer) v1._2;
            }
        });

        for (int index = 2; index < args.length; index++) {

            final Integer reqUserId = Integer.parseInt(args[index]);
            DeleteExistingOffers(reqUserId);

            JavaRDD<Tuple2<Object, Object>> itemsNotRatedByUser = notRatedByUser
                    .map(new Function<Integer, Tuple2<Object, Object>>() {
                        public Tuple2<Object, Object> call(Integer r) {
                            return new Tuple2<Object, Object>(reqUserId, r);
                        }
                    });
            System.out.println(itemsNotRatedByUser.collect().toString());
            JavaRDD<Rating> recomondations = model.predict(itemsNotRatedByUser.rdd()).toJavaRDD().distinct();
            recomondations = recomondations.sortBy(new Function<Rating, Double>() {
                public Double call(Rating v1) throws Exception {
                    return v1.rating();
                }

            }, false, 1);

            JavaRDD<Rating> topRecomondations = sc.parallelize(recomondations.take(10));

            // Join top 10 recommendations with item descriptions
            JavaRDD<Tuple2<Rating, String>> recommendedItems = topRecomondations.mapToPair(
                    new PairFunction<Rating, Integer, Rating>() {
                        public Tuple2<Integer, Rating> call(Rating t) throws Exception {
                            return new Tuple2<Integer, Rating>(t.product(), t);
                        }
                    }).join(itemDescritpion).values();

            PrintWriter pw = null;
            try {
                pw = new PrintWriter(new File("reco.csv"));
                recommendedItems.foreach(new VoidFunction<Tuple2<Rating, String>>() {
                    public void call(Tuple2<Rating, String> t) throws Exception {
                        System.out.println(t._1.user() + "\t" + t._1.rating() + "\t" + t._2);
                        UpdateNewOffers(t._1.user(), t._2, t._1.rating());
                    }
                });
                pw.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            /*try {
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
            } catch (Exception e) {
                e.printStackTrace();
            }*/
        }

        //JavaRDD<Rating> topRecomondations = sc.parallelize(recomondations.take(1));


        System.out.print("Successfull");

    }


    private synchronized static void DeleteExistingOffers(int userID) {

        Connection conn = null;
        try {
            Class.forName(sqlClassName);
            Properties props = new Properties();
            props.setProperty("user", user);
            props.setProperty("password", password);

            conn = DriverManager.getConnection(url, props);
            String delete_sql_statement = "DELETE from  recommendations " + "where  UserID=?";
            PreparedStatement preparedStatement = conn.prepareStatement(delete_sql_statement);
            preparedStatement.setInt(1, userID);
            System.out.println(preparedStatement);
            preparedStatement.executeUpdate();

        } catch (ClassNotFoundException e) {
            System.out.println("No MySQL JDBC Driver found");
            e.printStackTrace();
            return;
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }


    private synchronized static void UpdateNewOffers(int customerId, String mcc, double rating) {

        Connection conn = null;
        try {
            Class.forName(sqlClassName);
            Properties props = new Properties();
            props.setProperty("user", user);
            props.setProperty("password", password);
            conn = DriverManager.getConnection(url, props);

            String insert_sql_statement = "INSERT into recommendations (UserID, MCC, Rating)" + "VALUES (?, ?, ?)";
            PreparedStatement preparedStatement = conn.prepareStatement(insert_sql_statement);
            preparedStatement.setInt(1, customerId);
            preparedStatement.setString(2, mcc);
            preparedStatement.setDouble(3, rating);

            preparedStatement.executeUpdate();

        } catch (ClassNotFoundException e) {
            System.out.println("No MySQL JDBC Driver found");
            e.printStackTrace();
            return;
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

    }

}