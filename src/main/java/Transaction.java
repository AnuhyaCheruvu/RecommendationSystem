
import java.io.Serializable;
import java.util.Date;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class Transaction implements Serializable{

    private Long custId;
    
    private String cardholderLastName;
    
    private String cardHolderInitial;
    
    private String description;
    
    private Double amount;
    
    private String vendor;
    
    private Date transactionDate;

    private Long productId;
    
    private String mcc;

    public Transaction(Long custId, String cardholderLastName, String cardHolderInitial, String description, Double amount, String vendor, Date transactionDate, Long productId, String mcc) {
        this.custId = custId;
        this.cardholderLastName = cardholderLastName;
        this.cardHolderInitial = cardHolderInitial;
        this.description = description;
        this.amount = amount;
        this.vendor = vendor;
        this.transactionDate = transactionDate;
        this.productId = productId;
        this.mcc = mcc;
    }

    public String getCardholderLastName() {
        return cardholderLastName;
    }

    public void setCardholderLastName(String cardholderLastName) {
        this.cardholderLastName = cardholderLastName;
    }

    public String getCardHolderInitial() {
        return cardHolderInitial;
    }

    public void setCardHolderInitial(String cardHolderInitial) {
        this.cardHolderInitial = cardHolderInitial;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Double getAmount() {
        return amount;
    }

    public void setAmount(Double amount) {
        this.amount = amount;
    }

    public String getVendor() {
        return vendor;
    }

    public void setVendor(String vendor) {
        this.vendor = vendor;
    }

    public Date getTransactionDate() {
        return transactionDate;
    }

    public void setTransactionDate(Date transactionDate) {
        this.transactionDate = transactionDate;
    }

    public String getMcc() {
        return mcc;
    }

    public void setMcc(String mcc) {
        this.mcc = mcc;
    }

    public Long getCustId() {
        return custId;
    }

    public void setCustId(Long custId) {
        this.custId = custId;
    }

    public Long getProductId() {
        return productId;
    }

    public void setProductId(Long productId) {
        this.productId = productId;
    }

    @Override
    public String toString() {
        return "Transaction{" +
                "custId=" + custId +
                ", cardholderLastName='" + cardholderLastName + '\'' +
                ", cardHolderInitial='" + cardHolderInitial + '\'' +
                ", description='" + description + '\'' +
                ", amount=" + amount +
                ", vendor='" + vendor + '\'' +
                ", transactionDate=" + transactionDate +
                ", productId=" + productId +
                ", mcc='" + mcc + '\'' +
                '}';
    }
}
