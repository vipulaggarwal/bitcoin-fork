#include <omp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <algorithm>
#include <cstdlib> 

using namespace Rcpp;

arma::mat exogenous_creator(const arma::vec& prices,const arma::vec& fees,const int& s_firm,const int& eda_value,const double& btcdiff,const double& bchdiff,
                            const int& btc_lagged,const int& bch_lagged,const int& btc_lagged_other,const int& bch_lagged_other){
  
  arma::mat exogenous_matrix = arma::zeros(2,15);
  
  exogenous_matrix(0,0)=prices(0);
  exogenous_matrix(1,0)=prices(1);
  
  exogenous_matrix(0,1)=btcdiff;
  exogenous_matrix(1,1)=bchdiff;
  
  exogenous_matrix(0,2)=fees(0);
  exogenous_matrix(1,2)=fees(1);
  
  exogenous_matrix(0,3)=prices(0)-prices(1);
  exogenous_matrix(1,3)=prices(0)-prices(1);
  
  exogenous_matrix(0,4)=btcdiff-bchdiff;
  exogenous_matrix(1,4)=btcdiff-bchdiff;
  
  if(s_firm!=1){
    exogenous_matrix(0,(4+s_firm-1))=1;
    exogenous_matrix(1,(4+s_firm-1))=1;
  }
  
  exogenous_matrix(0,10)=eda_value;
  exogenous_matrix(1,10)=eda_value;
  
  exogenous_matrix(0,11)=eda_value*exogenous_matrix(0,3);
  exogenous_matrix(1,11)=eda_value*exogenous_matrix(1,3);
  
  exogenous_matrix(0,12)=eda_value*exogenous_matrix(0,4);
  exogenous_matrix(1,12)=eda_value*exogenous_matrix(1,4);
  
  exogenous_matrix(0,13)=btc_lagged;
  exogenous_matrix(1,13)=bch_lagged;
  
  exogenous_matrix(0,14)=btc_lagged_other;
  exogenous_matrix(1,14)=bch_lagged_other;
  
  return exogenous_matrix;
}

arma::mat counterfactual_exogenous_creator(const arma::vec& prices,const arma::vec& fees,const int& eda_value,const double& btcdiff,const double& bchdiff){
  
  arma::mat exogenous_matrix = arma::zeros(2,15);
  
  exogenous_matrix(0,0)=prices(0);
  exogenous_matrix(1,0)=prices(1);
  
  exogenous_matrix(0,1)=btcdiff;
  exogenous_matrix(1,1)=bchdiff;
  
  exogenous_matrix(0,2)=fees(0);
  exogenous_matrix(1,2)=fees(1);
  
  exogenous_matrix(0,3)=prices(0)-prices(1);
  exogenous_matrix(1,3)=prices(0)-prices(1);
  
  exogenous_matrix(0,4)=btcdiff-bchdiff;
  exogenous_matrix(1,4)=btcdiff-bchdiff;
  
  exogenous_matrix(0,10)=eda_value;
  exogenous_matrix(1,10)=eda_value;
  
  exogenous_matrix(0,11)=eda_value*exogenous_matrix(0,3);
  exogenous_matrix(1,11)=eda_value*exogenous_matrix(1,3);
  
  exogenous_matrix(0,12)=eda_value*exogenous_matrix(0,4);
  exogenous_matrix(1,12)=eda_value*exogenous_matrix(1,4);
  
  return exogenous_matrix;
}


  
arma::vec policy_function(const arma::rowvec& x1,const arma::rowvec& x2,const arma::vec& beta1, const arma::vec& beta2,const arma::vec& cutoffs1,
                          const arma::vec& cutoffs2, const double& gam, const double& error1,const double& error2){
  
  double y1 = arma::accu(arma::conv_to<arma::vec>::from(x1)%beta1) + error1;
  double y2 = arma::accu(arma::conv_to<arma::vec>::from(x2)%beta2) + gam*y1 + error2;
  arma::uvec btccount=arma::find(cutoffs1<y1);
  arma::uvec bchcount=arma::find(cutoffs2<y2);
  arma::vec blocks=arma::zeros(2);
  
  if(!btccount.is_empty()){
    blocks(0)=arma::as_scalar(btccount.tail(1))+1; // +1 because of 0 indexing in c++
  }
  
  if(!bchcount.is_empty()){
    blocks(1)=arma::as_scalar(bchcount.tail(1))+1;
  }
  
  return blocks;
}


arma::mat other_firms(const int& s_firm,const arma::rowvec& x1_,const arma::rowvec& x2_,const arma::vec& beta1, const arma::vec& beta2,
                      const arma::vec& cutoffs1,const arma::vec& cutoffs2, const double& gam, const arma::mat& error_i){
  
  arma::vec firms = arma::linspace<arma::vec>(1,6,6);
  firms=firms.elem(find(firms!=s_firm));
  arma::mat results = arma::zeros(5,4);
  int firm;
  arma::rowvec x1 = x1_;
  arma::rowvec x2 = x2_;
  
  for (int i=0;i<5;i++){
    firm=firms(i);
    x1(arma::span(5,9)).fill(0);
    x2(arma::span(5,9)).fill(0);
    if(firm!=1){
      x1(4+firm-1)=1;
      x2(4+firm-1)=1;
    }
    results.row(i).cols(0,1)=arma::conv_to<arma::rowvec>::from(policy_function(x1,x2,beta1,beta2,cutoffs1,cutoffs2,gam,error_i(i,0),error_i(i,1)));
    results.row(i).cols(2,3)=error_i.row(i);
  }
  
  return results;
}


arma::vec policy_function_uniformerror(const arma::rowvec& x1,const arma::rowvec& x2,const arma::vec& beta1, const arma::vec& beta2,const arma::vec& cutoffs1,
                            const arma::vec& cutoffs2, const double& gam, const double& error1,const double& error2){
    
  double y1 = arma::accu(arma::conv_to<arma::vec>::from(x1)%beta1);
  double y2 = arma::accu(arma::conv_to<arma::vec>::from(x2)%beta2);
  double prob1 = R::qnorm(error1, 0.0, 1.0, true, false);
  double prob2 = R::qnorm(error2, 0.0, 1.0, true, false);
  y1+=prob1;
  y2+=prob2;
  arma::uvec btccount=arma::find(cutoffs1<y1);
  arma::uvec bchcount=arma::find(cutoffs2<y2);
  arma::vec blocks=arma::zeros(2);
    
  if(!btccount.is_empty()){
    blocks(0)=arma::as_scalar(btccount.tail(1))+1; // +1 because of 0 indexing in c++
  }
    
  if(!bchcount.is_empty()){
    blocks(1)=arma::as_scalar(bchcount.tail(1))+1;
  }
  return blocks;
}


arma::mat other_firms_uniformerror(const int& s_firm,const arma::rowvec& x1_,const arma::rowvec& x2_,const arma::vec& beta1, const arma::vec& beta2,
                      const arma::vec& cutoffs1,const arma::vec& cutoffs2, const double& gam, const arma::mat& error_i){
  
  arma::vec firms = arma::linspace<arma::vec>(1,6,6);
  firms=firms.elem(find(firms!=s_firm));
  arma::mat results = arma::zeros(5,4);
  int firm;
  arma::rowvec x1 = x1_;
  arma::rowvec x2 = x2_;
  
  for (int i=0;i<5;i++){
    firm=firms(i);
    x1(arma::span(5,9)).fill(0);
    x2(arma::span(5,9)).fill(0);
    if(firm!=1){
      x1(4+firm-1)=1;
      x2(4+firm-1)=1;
    }
    results.row(i).cols(0,1)=arma::conv_to<arma::rowvec>::from(policy_function_uniformerror(x1,x2,beta1,beta2,cutoffs1,cutoffs2,gam,error_i(i,0),error_i(i,1)));
    results.row(i).cols(2,3)=error_i.row(i);
  }
  
  return results;
}

arma::vec policy_function_singleerror(const arma::rowvec& x1,const arma::rowvec& x2,const arma::vec& beta1, const arma::vec& beta2,const arma::vec& cutoffs1,
                          const arma::vec& cutoffs2, const double& gam, const double& error1){
  
  double y1 = arma::accu(arma::conv_to<arma::vec>::from(x1)%beta1) + error1;
  double y2 = arma::accu(arma::conv_to<arma::vec>::from(x2)%beta2) + gam*y1 + error1;
  arma::uvec btccount=arma::find(cutoffs1<y1);
  arma::uvec bchcount=arma::find(cutoffs2<y2);
  arma::vec blocks=arma::zeros(2);
  
  if(!btccount.is_empty()){
    blocks(0)=arma::as_scalar(btccount.tail(1))+1; // +1 because of 0 indexing in c++
  }
  
  if(!bchcount.is_empty()){
    blocks(1)=arma::as_scalar(bchcount.tail(1))+1;
  }
  
  return blocks;
}


arma::mat other_firms_singleerror(const int& s_firm,const arma::rowvec& x1_,const arma::rowvec& x2_,const arma::vec& beta1, const arma::vec& beta2,
                      const arma::vec& cutoffs1,const arma::vec& cutoffs2, const double& gam, const arma::vec& error_i){
  
  arma::vec firms = arma::linspace<arma::vec>(1,6,6);
  firms=firms.elem(find(firms!=s_firm));
  arma::mat results = arma::zeros(5,4);
  int firm;
  arma::rowvec x1 = x1_;
  arma::rowvec x2 = x2_;
  
  for (int i=0;i<5;i++){
    firm=firms(i);
    x1(arma::span(5,9)).fill(0);
    x2(arma::span(5,9)).fill(0);
    if(firm!=1){
      x1(4+firm-1)=1;
      x2(4+firm-1)=1;
    }
    results.row(i).cols(0,1)=arma::conv_to<arma::rowvec>::from(policy_function_singleerror(x1,x2,beta1,beta2,cutoffs1,cutoffs2,gam,error_i(i)));
    results(i,2)=error_i(i);
  }
  
  return results;
}


arma::mat firms_decisions(const arma::rowvec& x1_,const arma::rowvec& x2_,const arma::vec& beta1, const arma::vec& beta2,
                      const arma::vec& cutoffs1,const arma::vec& cutoffs2, const double& gam, const arma::mat& error_i,
                      const arma::mat& lagged,const arma::mat& lagged_other){
  
  arma::vec firms = arma::linspace<arma::vec>(1,6,6);
  arma::mat results = arma::zeros(6,4);
  int firm;
  arma::rowvec x1 = x1_;
  arma::rowvec x2 = x2_;
  
  for (int i=0;i<6;i++){
    
    firm=firms(i);
    x1(arma::span(5,9)).fill(0);
    x2(arma::span(5,9)).fill(0);
    x1(13) = lagged(0,i);
    x2(13) = lagged(0,i+6);
    x1(14) = lagged_other(0,i);
    x2(14) = lagged_other(0,i+6);
    
    if(firm!=1){
      x1(4+firm-1)=1;
      x2(4+firm-1)=1;
    }
    
    results.row(i).cols(0,1)=arma::conv_to<arma::rowvec>::from(policy_function(x1,x2,beta1,beta2,cutoffs1,cutoffs2,gam,error_i(i,0),error_i(i,1)));
    results.row(i).cols(2,3)=error_i.row(i);
  }
  
  return results;
}



double btc_diff_adjustment(const double& current_diff, const int& running_coins, const int& coins_current, const int& ts_last, 
                           const int& ts_current){
  
  double ts_diff;
  if((running_coins+coins_current)>= 2016) {
    ts_diff= (double)(ts_current - ts_last);
    
    if(ts_diff<14*86400*0.25){
      ts_diff=14*86400*0.25;
    }
    if(ts_diff>14*86400*4){
      ts_diff=14*86400*4;
    }
    return (current_diff-std::log(ts_diff/(14*86400)));
    
  } else {
    
    return current_diff;
  }
}


Rcpp::List bch_queue_update(const List& bch_queue,const int& bch_coin, const int& interval, const int& eda, const double& diff){
  
  int sum_bch=0;
  List bch_queue_temp=clone(bch_queue);
  std::deque<double> q1_copy, q2_copy, q3_copy;
  std::deque<double> q1 = as<std::deque<double>>(as<NumericVector>(bch_queue_temp[0]));
  std::deque<double> q2 = as<std::deque<double>>(as<NumericVector>(bch_queue_temp[1]));
  std::deque<double> q3 = as<std::deque<double>>(as<NumericVector>(bch_queue_temp[2]));
  
  if(eda==1){
    
    if(bch_coin>0){
      
      q1.push_back(interval);
      q2.push_back(bch_coin);
      q3.push_back(diff);
      q2_copy = q2;
      q2_copy.pop_front();
      sum_bch=std::accumulate(q2_copy.begin(),q2_copy.end(),0);
      if(sum_bch>156){ //we use 156 instead of 20 due to the edge case of time period close to end of eda period
        q1.pop_front();
        q2.pop_front();
        q3.pop_front();
      }
      
    }
    NumericVector v1(q1.begin(),q1.end());
    bch_queue_temp[0] = as<NumericVector>(v1);
    NumericVector v2(q2.begin(),q2.end());
    bch_queue_temp[1] = as<NumericVector>(v2);
    NumericVector v3(q3.begin(),q3.end());
    bch_queue_temp[2] = as<NumericVector>(v3);
    return bch_queue_temp;
    
  } else {
    
    if(bch_coin>0){
      q1.push_back(interval);
      q2.push_back(bch_coin);
      q3.push_back(diff);
      q2_copy = q2;
      q2_copy.pop_front();
      sum_bch=std::accumulate(q2_copy.begin(),q2_copy.end(),0);
      if(sum_bch>156){
        q1.pop_front();
        q2.pop_front();
        q3.pop_front();
      }
      
    }
    
    NumericVector v1(q1.begin(),q1.end());
    bch_queue_temp[0] = as<NumericVector>(v1);
    NumericVector v2(q2.begin(),q2.end());
    bch_queue_temp[1] = as<NumericVector>(v2);
    NumericVector v3(q3.begin(),q3.end());
    bch_queue_temp[2] = as<NumericVector>(v3);
    return bch_queue_temp;
    
  }
  
}


double bch_diff_adjustment(const double& current_diff, const int& running_coins, const int& coins_current, const int& ts_last, 
                           const int& ts_current, const int& eda,const List& bch_queue, int& mechanism){
  
  if(coins_current==0){
    return current_diff;
  }
  
  int ind1,ind2;
  double ts_diff;
  List bch_queue_temp=clone(bch_queue);
  
  arma::vec q1 = arma::reverse(as<arma::vec>(bch_queue_temp[0])); //time intervals
  arma::vec q2 = arma::reverse(as<arma::vec>(bch_queue_temp[1])); // bch coins
  arma::uvec ps1,ps2;
  
  if(eda==1){
    
    ps1=arma::find(arma::cumsum(q2)>=6);
    ps2=arma::find(arma::cumsum(q2)>=12);
    
    if(ps1.is_empty() || ps2.is_empty()){
      return current_diff;
    }
    
    ind1 = ps1(0);
    ind2 = ps2(0);
    
    ts_diff=(double)q1(ind1)-q1(ind2);
    
    if(ts_diff>43200){
      
      mechanism=0;
      return current_diff + std::log(0.8);
      
    } else if(running_coins+coins_current>=2016){
      
      ts_diff=(double)ts_current-ts_last;
      
      if(ts_diff<14*86400*0.25){
        ts_diff=14*86400*0.25;
      }
      
      if(ts_diff>14*86400*4){
        ts_diff=14*86400*4;
      }
      
      mechanism=1;
      return current_diff-std::log((ts_diff/(14*86400)));
      
    } else {
      
      return current_diff;
    }
  } else {
    
    arma::vec q3 = arma::reverse(as<arma::vec>(bch_queue_temp[2])); // bch difficulty
    double workdone=0.0;
    ps1=arma::find(arma::cumsum(q2)>=2);
    ps2=arma::find(arma::cumsum(q2)>=145);
    
    if(ps1.is_empty() || ps2.is_empty()){
      return current_diff;
    }
    
    ind1 = ps1(0);
    ind2 = ps2(0);
    
    ts_diff=(double)q1(ind1)-q1(ind2);
    workdone=arma::mean(q3(arma::span(ind1,ind2)));
    
    if(ts_diff<86400*0.5){
      ts_diff=86400*0.5;
    }
    if(ts_diff>86400*2){
      ts_diff=86400*2;
    }
    
    mechanism=2;
    return current_diff-std::log((ts_diff/(86400)))-current_diff+workdone;
  }
  
}

/*
arma::vec profits_precompute(const arma::vec& decisions,const arma::rowvec& error, const arma::mat& other_firms_decisions, const arma::vec& prices, const arma::vec&fees, const int& s_firm
                               , const int& eda, const double& btcdiff, const double& bchdiff, const int& t, const int& btc_lagged, const int& bch_lagged,
                               const arma::mat& btc_cum,const arma::mat& bch_cum, const arma::uvec& index_cumul, const int& compute_col){
  
  arma::vec results = arma::zeros(compute_col);
  double discount_factor=std::pow(0.9,t);
  arma::vec btc_cumul = arma::vectorise(btc_cum);
  arma::vec bch_cumul = arma::vectorise(bch_cum);
  
  int btc_i=decisions(0);
  int bch_i=decisions(1);
  
  int btc_other=arma::sum(other_firms_decisions.col(0));
  int bch_other=arma::sum(other_firms_decisions.col(1));
  
  btc_cumul(index_cumul) += other_firms_decisions.col(0);
  bch_cumul(index_cumul) += other_firms_decisions.col(1);
  arma::vec firms = arma::zeros(5);
  if(s_firm!=1){
    firms(s_firm-2)=1;
  }
  
  arma::vec vec_btc = arma::zeros(22);
  arma::vec vec_bch = arma::zeros(22);
  
  vec_btc(0)=prices(0);
  vec_btc(1)=fees(0);
  vec_btc(2)=-btcdiff;
  vec_btc(3)=btc_i*12.5;
  vec_btc(4)=-btc_other*12.5;
  vec_btc(12)=btc_cumul(s_firm-1);
  vec_btc(13)=-arma::sum(btc_cumul(index_cumul));
  vec_btc(16)=error(0);
  vec_btc(arma::span(17,21))=firms;
  
  if(btc_i!=0){
    
    vec_btc(5) = -(btcdiff + std::log(btc_i));
    vec_btc(6) = prices(0) + std::log(btc_i*12.5);
    vec_btc(8) = fees(0) + std::log(btc_i);
    vec_btc(9) = fees(0) + std::log(btc_i) + prices(0);
    vec_btc(14) = std::log(btc_i*12.5);
    
    if(btc_other!=0){
      
      vec_btc(7) = -(prices(0) + std::log(btc_other*12.5));
      vec_btc(10) = -(fees(0) + std::log(btc_other));
      vec_btc(11) = -(fees(0) + std::log(btc_other) + prices(0));
      vec_btc(15) = -std::log(btc_other*12.5);
      
    }
    
  } else {
    
    vec_btc*=0;
    
  }
  
  vec_bch(0)=prices(1);
  vec_bch(1)=fees(1);
  vec_bch(2)=-bchdiff;
  vec_bch(3)=bch_i*12.5;
  vec_bch(4)=-bch_other*12.5;
  vec_bch(12)=bch_cumul(s_firm-1);
  vec_bch(13)=-arma::sum(bch_cumul(index_cumul));
  vec_bch(16)=error(1);
  vec_bch(arma::span(17,21))=firms;
  
  if(bch_i!=0){
    
    vec_bch(5) = -(bchdiff + std::log(bch_i));
    vec_bch(6) = prices(1) + std::log(bch_i*12.5);
    vec_bch(8) = fees(1) + std::log(bch_i);
    vec_bch(9) = fees(1) + std::log(bch_i) + prices(1);
    vec_bch(14) = std::log(bch_i*12.5);
    
    if(bch_other!=0){
      
      vec_bch(7) = -(prices(1) + std::log(bch_other*12.5));
      vec_bch(10) = -(fees(1) + std::log(bch_other));
      vec_bch(11) = -(fees(1) + std::log(bch_other) + prices(1));
      vec_bch(15) = -std::log(bch_other*12.5);
      
    }
    
  } else {
    
    vec_bch*=0;
    
  }
  
  results(arma::span(0,21)) = vec_btc;
  results(arma::span(22,43)) = vec_bch;

  results(44) = eda;
  results(45) = (eda==1 && (bch_i+bch_other)<=1) ? 1 : 0;
  results(46) = (eda==1 && (bch_i+bch_other)==0) ? 1 : 0;
  results(47) = (eda==1 && btc_i==0 && btc_lagged>0 && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(48) = (eda==1 && btc_i < btc_lagged && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;

  return results*discount_factor;
}
*/

arma::vec profits_precompute(const arma::vec& decisions,const arma::rowvec& error, const arma::mat& other_firms_decisions, const arma::vec& prices, const arma::vec&fees, const int& s_firm
                               , const int& eda, const double& btcdiff, const double& bchdiff, const int& t, const int& btc_lagged, const int& bch_lagged,
                               const arma::mat& btc_cum,const arma::mat& bch_cum, const arma::uvec& index_cumul, const int& compute_col){
  
  arma::vec results = arma::zeros(compute_col);
  double discount_factor=std::pow(0.9,t);
  arma::vec btc_cumul = arma::vectorise(btc_cum);
  arma::vec bch_cumul = arma::vectorise(bch_cum);
  
  double pdbtc,pdbch;
  pdbtc = std::log(12.5+std::exp(fees(0))) + prices(0) - btcdiff; //dari
  pdbch = std::log(12.5+std::exp(fees(1))) + prices(1) - bchdiff;
  
  int btc_i=decisions(0);
  int bch_i=decisions(1);
  
  double costfactor = std::pow(2,32)/(1800*14*std::pow(10,12));
  costfactor =costfactor*0.04*0.5*1.5;
  
  int btc_other=arma::sum(other_firms_decisions.col(0));
  int bch_other=arma::sum(other_firms_decisions.col(1));
  
  btc_cumul(index_cumul) += other_firms_decisions.col(0);
  bch_cumul(index_cumul) += other_firms_decisions.col(1);
  arma::vec firms = arma::zeros(5);
  if(s_firm!=1){
    firms(s_firm-2)=1;
  }
  
  arma::vec vec_btc = arma::zeros(20);
  arma::vec vec_bch = arma::zeros(20);
  
  vec_btc(0)=std::exp(prices(0));
  vec_btc(1)=std::exp(fees(0));
  vec_btc(2)=-std::exp(btcdiff);
  vec_btc(3)=btc_i*12.5;
  vec_btc(4)=-btc_other*12.5;
  vec_btc(12)=btc_cumul(s_firm-1);
  vec_btc(13)=-arma::sum(btc_cumul(index_cumul));
  vec_btc(16)=error(0);
  vec_btc(17)=prices(0);
  vec_btc(18)=fees(0);
  vec_btc(19)=-btcdiff;
  
  if(btc_i!=0){
    
    vec_btc(5) = -std::exp(btcdiff + std::log(btc_i))*costfactor;
    vec_btc(6) = std::exp(prices(0) + std::log(btc_i*12.5));
    vec_btc(8) = std::exp(fees(0) + std::log(btc_i));
    vec_btc(9) = std::exp(fees(0) + std::log(btc_i) + prices(0));
    vec_btc(14) = std::exp(std::log(btc_i*12.5));
    
    if(btc_other!=0){
      
      vec_btc(7) = -std::exp(prices(0) + std::log(btc_other*12.5));
      vec_btc(10) = -std::exp(fees(0) + std::log(btc_other));
      vec_btc(11) = -std::exp(fees(0) + std::log(btc_other) + prices(0));
      vec_btc(15) = -std::exp(std::log(btc_other*12.5));
      
    }
    
  } else {
    
    vec_btc*=0;
    
  }
  
  vec_bch(0)=std::exp(prices(1));
  vec_bch(1)=std::exp(fees(1));
  vec_bch(2)=-std::exp(bchdiff);
  vec_bch(3)=bch_i*12.5;
  vec_bch(4)=-bch_other*12.5;
  vec_bch(12)=bch_cumul(s_firm-1);
  vec_bch(13)=-arma::sum(bch_cumul(index_cumul));
  vec_bch(16)=error(1);
  vec_bch(17)=prices(1);
  vec_bch(18)=fees(1);
  vec_bch(19)=-bchdiff;
  
  if(bch_i!=0){
    
    vec_bch(5) = -std::exp(bchdiff + std::log(bch_i))*costfactor;
    vec_bch(6) = std::exp(prices(1) + std::log(bch_i*12.5));
    vec_bch(8) = std::exp(fees(1) + std::log(bch_i));
    vec_bch(9) = std::exp(fees(1) + std::log(bch_i) + prices(1));
    vec_bch(14) = std::exp(std::log(bch_i*12.5));
    
    if(bch_other!=0){
      
      vec_bch(7) = -std::exp(prices(1) + std::log(bch_other*12.5));
      vec_bch(10) = -std::exp(fees(1) + std::log(bch_other));
      vec_bch(11) = -std::exp(fees(1) + std::log(bch_other) + prices(1));
      vec_bch(15) = -std::exp(std::log(bch_other*12.5));
      
    }
    
  } else {
    
    vec_bch*=0;
    
  }
  
  results(arma::span(0,19)) = vec_btc;
  results(arma::span(20,39)) = vec_bch;
  if(btc_i!=0 || bch_i!=0){
    results(arma::span(40,44))=firms;
  }
  results(45) = eda;
  results(46) = (eda==1 && (bch_i+bch_other)<=1) ? 1 : 0;
  results(47) = (eda==1 && (bch_i+bch_other)<=1  && (btc_i==0)) ? 1 : 0;
  results(48) = (eda==1 && (bch_i+bch_other)<=1  && (pdbtc >= pdbch)) ? 1 : 0;
  results(49) = (eda==1 && btc_i==0 && btc_lagged>0 && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(50) = (eda==1 && btc_i==0 && btc_lagged>=0 && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(51) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(52) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0 && pdbch>pdbtc) ? 1 : 0;
  results(53) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0 && pdbch<=pdbtc) ? 1 : 0;
  
  return results*discount_factor;
}



arma::vec profits_precompute_singleerror(const arma::vec& decisions,const double& error, const arma::mat& other_firms_decisions, const arma::vec& prices, const arma::vec&fees, const int& s_firm
                               , const int& eda, const double& btcdiff, const double& bchdiff, const int& t, const int& btc_lagged, const int& bch_lagged,
                               const arma::mat& btc_cum,const arma::mat& bch_cum, const arma::uvec& index_cumul, const int& compute_col){
  
  arma::vec results = arma::zeros(compute_col);
  double discount_factor=std::pow(0.9,t);
  arma::vec btc_cumul = arma::vectorise(btc_cum);
  arma::vec bch_cumul = arma::vectorise(bch_cum);
  
  double pdbtc,pdbch;
  pdbtc = std::log(12.5+std::exp(fees(0))) + prices(0) - btcdiff; //dari
  pdbch = std::log(12.5+std::exp(fees(1))) + prices(1) - bchdiff;
  
  int btc_i=decisions(0);
  int bch_i=decisions(1);
  
  double costfactor = std::pow(2,32)/(1800*14*std::pow(10,12));
  costfactor =costfactor*0.04*0.5*1.5;
  
  int btc_other=arma::sum(other_firms_decisions.col(0));
  int bch_other=arma::sum(other_firms_decisions.col(1));
  
  btc_cumul(index_cumul) += other_firms_decisions.col(0);
  bch_cumul(index_cumul) += other_firms_decisions.col(1);
  arma::vec firms = arma::zeros(5);
  if(s_firm!=1){
    firms(s_firm-2)=1;
  }
  
  arma::vec vec_btc = arma::zeros(20);
  arma::vec vec_bch = arma::zeros(20);
  
  vec_btc(0)=std::exp(prices(0));
  vec_btc(1)=std::exp(fees(0));
  vec_btc(2)=-std::exp(btcdiff);
  vec_btc(3)=btc_i*12.5;
  vec_btc(4)=-btc_other*12.5;
  vec_btc(12)=btc_cumul(s_firm-1);
  vec_btc(13)=-arma::sum(btc_cumul(index_cumul));
  vec_btc(16)=error;
  vec_btc(17)=prices(0);
  vec_btc(18)=fees(0);
  vec_btc(19)=-btcdiff;
  
  if(btc_i!=0){
    
    vec_btc(5) = -std::exp(btcdiff + std::log(btc_i))*costfactor;
    vec_btc(6) = std::exp(prices(0) + std::log(btc_i*12.5));
    vec_btc(8) = std::exp(fees(0) + std::log(btc_i));
    vec_btc(9) = std::exp(fees(0) + std::log(btc_i) + prices(0));
    vec_btc(14) = std::exp(std::log(btc_i*12.5));
    
    if(btc_other!=0){
      
      vec_btc(7) = -std::exp(prices(0) + std::log(btc_other*12.5));
      vec_btc(10) = -std::exp(fees(0) + std::log(btc_other));
      vec_btc(11) = -std::exp(fees(0) + std::log(btc_other) + prices(0));
      vec_btc(15) = -std::exp(std::log(btc_other*12.5));
      
    }
    
  } else {
    
    vec_btc*=0;
    
  }
  
  vec_bch(0)=std::exp(prices(1));
  vec_bch(1)=std::exp(fees(1));
  vec_bch(2)=-std::exp(bchdiff);
  vec_bch(3)=bch_i*12.5;
  vec_bch(4)=-bch_other*12.5;
  vec_bch(12)=bch_cumul(s_firm-1);
  vec_bch(13)=-arma::sum(bch_cumul(index_cumul));
  vec_bch(16)=error;
  vec_bch(17)=prices(1);
  vec_bch(18)=fees(1);
  vec_bch(19)=-bchdiff;
  
  if(bch_i!=0){
    
    vec_bch(5) = -std::exp(bchdiff + std::log(bch_i))*costfactor;
    vec_bch(6) = std::exp(prices(1) + std::log(bch_i*12.5));
    vec_bch(8) = std::exp(fees(1) + std::log(bch_i));
    vec_bch(9) = std::exp(fees(1) + std::log(bch_i) + prices(1));
    vec_bch(14) = std::exp(std::log(bch_i*12.5));
    
    if(bch_other!=0){
      
      vec_bch(7) = -std::exp(prices(1) + std::log(bch_other*12.5));
      vec_bch(10) = -std::exp(fees(1) + std::log(bch_other));
      vec_bch(11) = -std::exp(fees(1) + std::log(bch_other) + prices(1));
      vec_bch(15) = -std::exp(std::log(bch_other*12.5));
      
    }
    
  } else {
    
    vec_bch*=0;
    
  }
  
  results(arma::span(0,19)) = vec_btc;
  results(arma::span(20,39)) = vec_bch;
  if(btc_i!=0 || bch_i!=0){
    results(arma::span(40,44))=firms;
    results(54)=error;
  }
  results(45) = eda;
  results(46) = (eda==1 && (bch_i+bch_other)<=1) ? 1 : 0;
  results(47) = (eda==1 && (bch_i+bch_other)<=1  && (btc_i==0)) ? 1 : 0;
  results(48) = (eda==1 && (bch_i+bch_other)<=1  && (pdbtc >= pdbch)) ? 1 : 0;
  results(49) = (eda==1 && btc_i==0 && btc_lagged>0 && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(50) = (eda==1 && btc_i==0 && btc_lagged>=0 && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(51) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0) ? 1 : 0;
  results(52) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0 && pdbch>pdbtc) ? 1 : 0;
  results(53) = (eda==1 && btc_i <= btc_lagged && bch_i>=bch_lagged && bch_i>0 && pdbch<=pdbtc) ? 1 : 0;

  return results*discount_factor;
}

arma::vec counterfactual_profits(const arma::mat& all_firms_decisions, const arma::vec& prices, const arma::vec&fees,
                                const double& btcdiff, const double& bchdiff){
  
  arma::vec results = arma::zeros(24);
  results(arma::span(0,5))=all_firms_decisions.col(0);
  results(arma::span(6,11))=all_firms_decisions.col(1);

  double costfactor = std::pow(2,32)/(1800*14*std::pow(10,12));
  costfactor =costfactor*0.04*0.5*1.5;

  arma::vec vec_btc = arma::zeros(6);
  arma::vec vec_bch = arma::zeros(6);
  
  if(results(0)!=0){
    vec_btc(0)= std::exp(prices(0))*results(0)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(0)))*costfactor;
  }
  if(results(1)!=0){
    vec_btc(1)= std::exp(prices(0))*results(1)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(1)))*costfactor;
  }
  if(results(2)!=0){
    vec_btc(2)= std::exp(prices(0))*results(2)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(2)))*costfactor;
  }
  if(results(3)!=0){
    vec_btc(3)= std::exp(prices(0))*results(3)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(3)))*costfactor;
  }
  if(results(4)!=0){
    vec_btc(4)= std::exp(prices(0))*results(4)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(4)))*costfactor;
  }
  if(results(5)!=0){
    vec_btc(5)= std::exp(prices(0))*results(5)*(std::exp(fees(0))+12.5) - std::exp(btcdiff + std::log(results(5)))*costfactor;
  }
  
  if(results(6)!=0){
    vec_bch(0)= std::exp(prices(1))*results(6)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(6)))*costfactor;
  }
  if(results(7)!=0){
    vec_bch(1)= std::exp(prices(1))*results(7)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(7)))*costfactor;
  }
  if(results(8)!=0){
    vec_bch(2)= std::exp(prices(1))*results(8)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(8)))*costfactor;
  }
  if(results(9)!=0){
    vec_bch(3)= std::exp(prices(1))*results(9)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(9)))*costfactor;
  }
  if(results(10)!=0){
    vec_bch(4)= std::exp(prices(1))*results(10)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(10)))*costfactor;
  }
  if(results(11)!=0){
    vec_bch(5)= std::exp(prices(1))*results(11)*(std::exp(fees(1))+12.5) - std::exp(bchdiff + std::log(results(11)))*costfactor;
  }
  
  results(arma::span(12,17))=vec_btc;
  results(arma::span(18,23))=vec_bch;
  
  return results;
}


// [[Rcpp::export]]
arma::mat simulation_run(int s_firm, int num_pol, int compute_col,int interval, SEXP price_array, SEXP fee_array, SEXP firm_vec, int btc_difficulty_count,
                                       int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                       int btc_lagged, int bch_lagged,int btc_lagged_other, int bch_lagged_other, SEXP eda_vec, SEXP btc_cum, SEXP bch_cum, SEXP index_cum,
                                       int S, int T, SEXP actual_policy_sexp,SEXP alternate_policies_list_sexp, SEXP queue_list_sexp, int queue_index){
  
  
  arma::cube prices_array = as<arma::cube>(price_array);
  arma::cube fees_array = as<arma::cube>(fee_array);
  arma::vec firm_vector = as<arma::vec>(firm_vec);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_cumul = as<arma::vec>(btc_cum);
  arma::vec bch_cumul = as<arma::vec>(bch_cum);
  arma::uvec index_cumul = as<arma::uvec>(index_cum);
  Function rmvnorm = Environment("package:mvtnorm")["rmvnorm"];
  List actual_policy(actual_policy_sexp);
  List alternate_policies_list(alternate_policies_list_sexp);
  List queue_list(queue_list_sexp);
  //Rcpp::Rcout<<"Finished Loading."<<std::endl;
  NumericMatrix correlation(2,2);
  correlation(0,0)=correlation(1,1)=1.0;
  correlation(0,1)=correlation(1,0)=std::tanh(as<double>(actual_policy[4]));
  NumericVector mean_vec=NumericVector::create(0,0);
  /*
  arma::field<arma::cube> results(num_pol+1);
  for(int i =0;i<=num_pol;i++){
    arma::cube inst=arma::zeros(S,T,compute_col);
    results(i)=inst;
  }
  */
  arma::mat results = arma::zeros(num_pol+1,compute_col);

  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  arma::uvec row_ind(1);
  int mechanism;
  
  arma::mat diff_matrix = arma::zeros(num_pol+1,2);
  arma::mat running_matrix = arma::zeros(num_pol+1,2);
  arma::mat diff_change_matrix = arma::zeros(num_pol+1,2);
  arma::mat ts_current_matrix = arma::zeros(num_pol+1,2);
  arma::mat lagged_matrix = arma::zeros(num_pol+1,2);
  arma::mat lagged_other_matrix = arma::zeros(num_pol+1,2);
  arma::mat cumulative_matrix = arma::zeros(num_pol+1,12);
  NumericMatrix error_i_nm(T,2);
  NumericMatrix error_other_nm(T*5,2);
  List bch_eda_queue(num_pol+1);
  arma::mat ex_mt = arma::zeros(2,15);;
  arma::rowvec x1 = arma::zeros<arma::rowvec>(15);
  arma::rowvec x2 = arma::zeros<arma::rowvec>(15);
  arma::vec decisions_actual_i = arma::zeros(2);
  arma::mat other_firms_decisions = arma::zeros(5,4);
  arma::vec decisions_modified = arma::zeros(2);
  for(int s=0;s<S;s++){

    diff_matrix.fill(0);
    diff_matrix.col(0).fill(btc_diff);
    diff_matrix.col(1).fill(bch_diff);
    
    running_matrix.fill(0);
    running_matrix.col(0).fill(btc_difficulty_count);
    running_matrix.col(1).fill(bch_difficulty_count);
    
    diff_change_matrix.fill(0);
    diff_change_matrix.col(0).fill(btc_difficulty_change);
    diff_change_matrix.col(1).fill(bch_difficulty_change);
    
    ts_current_matrix.fill(0);
    ts_current_matrix.col(0).fill(interval);
    ts_current_matrix.col(1).fill(interval);
    
    lagged_matrix.fill(0);
    lagged_matrix.col(0).fill(btc_lagged);
    lagged_matrix.col(1).fill(bch_lagged);
    
    lagged_other_matrix.fill(0);
    lagged_other_matrix.col(0).fill(btc_lagged_other);
    lagged_other_matrix.col(1).fill(bch_lagged_other);
    
    cumulative_matrix.fill(0);
    cumulative_matrix.cols(0,5).each_row() = arma::conv_to<arma::rowvec>::from(btc_cumul);
    cumulative_matrix.cols(6,11).each_row() = arma::conv_to<arma::rowvec>::from(bch_cumul);
    
    std::fill(error_i_nm.begin(),error_i_nm.end(),0);
    error_i_nm = rmvnorm(T,mean_vec,correlation);
    arma::mat error_i(error_i_nm.begin(),T,2,false);
    
    std::fill(error_other_nm.begin(),error_other_nm.end(),0);
    error_other_nm = rmvnorm(T*5,mean_vec,correlation);
    arma::mat error_other(error_other_nm.begin(),T*5,2,false);
    
    for(int i=0;i<=num_pol;i++){
      
      bch_eda_queue[i]=Rcpp::clone(as<List>(queue_list[queue_index-1]));
    }
    
    for( int t=0;t<T;t++){
      //cout<<"t: "<<t<<endl;
      //print(bch_eda_queue[0]);
      
      ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                        s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),lagged_matrix(0,0),lagged_matrix(0,1),lagged_other_matrix(0,0),lagged_other_matrix(0,1));
      x1 = ex_mt.row(0);
      x2 = ex_mt.row(1);
      //cout<<"Exog"<<endl;
      //ex_mt.print();
      decisions_actual_i = policy_function(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                       ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_i(t,0),error_i(t,1));
      //cout<<"Decisions by i"<<endl;
      //decisions_actual_i.print();
      
      other_firms_decisions = other_firms(s_firm,x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                      ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_other(arma::span(5*t,5*(t+1)-1),arma::span::all));
      
      //cout<<"Decisions by other"<<endl;
      //other_firms_decisions.print();
      /*
      results(0)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute(decisions_actual_i,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),t,lagged_matrix(0,0),lagged_matrix(0,1),
              cumulative_matrix(arma::span(0),arma::span(0,5)),cumulative_matrix(arma::span(0),arma::span(6,11)),index_cumul,compute_col);
      */
      results.row(0) += arma::conv_to<arma::rowvec>::from(profits_precompute(decisions_actual_i,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),t,lagged_matrix(0,0),lagged_matrix(0,1),
              cumulative_matrix(arma::span(0),arma::span(0,5)),cumulative_matrix(arma::span(0),arma::span(6,11)),index_cumul,compute_col));
      
      
      //cout<<"profits vector"<<endl;
      //results(0)(span(s),span(t),span::all).print();
      
      total_btc_act=decisions_actual_i(0)+arma::sum(other_firms_decisions.col(0));
      total_bch_act=decisions_actual_i(1)+arma::sum(other_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(0,0)=decisions_actual_i(0);
      lagged_other_matrix(0,0)=sum(other_firms_decisions.col(0));
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),eda_vector(t), q_item_temp,mechanism);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism);
        
      }
      //cout<<"bch"<<endl;
      //cout<<total_bch_act<<endl;
      //cout<<"bch_diff"<<endl;
      //cout<<diff_matrix(0,1)<<endl;
      
      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)<1503428400){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)>=1503428400){
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(0,1)=decisions_actual_i(1);
      lagged_other_matrix(0,1)=sum(other_firms_decisions.col(1));
      
      ts_current_matrix.row(0)+=1800;
      
      row_ind<<0;
      
      cumulative_matrix(0,s_firm-1)=cumulative_matrix(0,s_firm-1)+decisions_actual_i(0);
      cumulative_matrix(0,s_firm-1+6)=cumulative_matrix(0,s_firm-1+6)+decisions_actual_i(1);
      
      cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
      cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
      
      for(int p =0;p<num_pol;p++){
        
        List new_policy(alternate_policies_list[p]);
        
        ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                          s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                                          lagged_other_matrix(p+1,0),lagged_other_matrix(p+1,1));
        x1 = ex_mt.row(0);
        x2 = ex_mt.row(1);
        
        decisions_modified = policy_function(x1,x2,as<arma::vec>(new_policy[0]),as<arma::vec>(new_policy[1])
                                                         ,as<arma::vec>(new_policy[2]),as<arma::vec>(new_policy[3]),new_policy[5],error_i(t,0),error_i(t,1));
        /*
        results(p+1)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute(decisions_modified,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
                arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),t,lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                cumulative_matrix(arma::span(p+1),arma::span(0,5)),cumulative_matrix(arma::span(p+1),arma::span(6,11)),index_cumul,compute_col);
        */
        
        results.row(p+1) += arma::conv_to<arma::rowvec>::from(profits_precompute(decisions_modified,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
                arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),t,lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                cumulative_matrix(arma::span(p+1),arma::span(0,5)),cumulative_matrix(arma::span(p+1),arma::span(6,11)),index_cumul,compute_col));
        
        total_btc_alt=decisions_modified(0)+arma::sum(other_firms_decisions.col(0));
        total_bch_alt=decisions_modified(1)+arma::sum(other_firms_decisions.col(1));
        
        new_btc_diff = btc_diff_adjustment(diff_matrix(p+1,0), running_matrix(p+1,0), total_btc_alt, diff_change_matrix(p+1,0), ts_current_matrix(p+1,0));
        
        if(new_btc_diff!=diff_matrix(p+1,0)){
          diff_change_matrix(p+1,0)=ts_current_matrix(p+1,0);
          diff_matrix(p+1,0)=new_btc_diff;
        }
        
        running_matrix(p+1,0)=((int)running_matrix(p+1,0)+total_btc_alt)%2016;
        lagged_matrix(p+1,0)=decisions_modified(0);
        lagged_other_matrix(p+1,0)=sum(other_firms_decisions.col(0));
        
        temp_bch = diff_matrix(p+1,1);
        mechanism=0;
        
        if(eda_vector(t)==1 && total_bch_alt>0 && total_bch_alt<15){
          
          for(int m=1;m<=total_bch_alt;m++){
            
            q_item_temp=bch_eda_queue[p+1];
            q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(p+1,1),eda_vector(t),temp_bch);
            new_diff=bch_diff_adjustment(temp_bch, running_matrix(p+1,1), m, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1),eda_vector(t), q_item_temp,mechanism);
            
            if(new_diff>temp_bch){
              temp_bch=new_diff;
              break;
            }
            
            temp_bch=new_diff;
          }
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff = temp_bch;
          
        } else {
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff=bch_diff_adjustment(diff_matrix(p+1,1),running_matrix(p+1,1), total_bch_alt, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1), eda_vector(t), bch_eda_queue[p+1],mechanism);
          
        }
        
        if(new_bch_diff == diff_matrix(p+1,1)){
          
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism!=0){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)<1503428400){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = total_bch_alt;
        } 
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)>=1503428400){
          
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        lagged_matrix(p+1,1)=decisions_modified(1);
        lagged_other_matrix(p+1,1)=sum(other_firms_decisions.col(1));
        
        ts_current_matrix.row(p+1)+=1800;
        
        row_ind<<p+1;
        
        cumulative_matrix(p+1,s_firm-1)=cumulative_matrix(p+1,s_firm-1)+decisions_modified(0);
        cumulative_matrix(p+1,s_firm-1+6)=cumulative_matrix(p+1,s_firm-1+6)+decisions_modified(1);
        
        cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
        cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
        
      }
      
    }
    
    
  }
  
  return results;
  
}

// [[Rcpp::export]]
arma::field<arma::cube> simulation_run_tdmedthod(int s_firm, int num_pol, int compute_col,int interval, SEXP price_array, SEXP fee_array, SEXP firm_vec, int btc_difficulty_count,
                                       int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                       int btc_lagged, int bch_lagged,int btc_lagged_other, int bch_lagged_other, SEXP eda_vec, SEXP btc_cum, SEXP bch_cum, SEXP index_cum,
                                       int S, int T, SEXP actual_policy_sexp, SEXP queue_list_sexp, int queue_index){
  
  
  arma::cube prices_array = as<arma::cube>(price_array);
  arma::cube fees_array = as<arma::cube>(fee_array);
  arma::vec firm_vector = as<arma::vec>(firm_vec);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_cumul = as<arma::vec>(btc_cum);
  arma::vec bch_cumul = as<arma::vec>(bch_cum);
  arma::uvec index_cumul = as<arma::uvec>(index_cum);
  Function rmvnorm = Environment("package:mvtnorm")["rmvnorm"];
  List actual_policy(actual_policy_sexp);
  List queue_list(queue_list_sexp);
  
  NumericMatrix correlation(2,2);
  correlation(0,0)=correlation(1,1)=1.0;
  correlation(0,1)=correlation(1,0)=std::tanh(as<double>(actual_policy[4]));
  NumericVector mean_vec=NumericVector::create(0,0);
  arma::field<arma::cube> results(num_pol+1);
  
  for(int i =0;i<=num_pol;i++){
    arma::cube inst=arma::zeros(S,T,compute_col);
    results(i)=inst;
  }
  
  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  arma::uvec row_ind(1);
  int mechanism;

  for(int s=0;s<S;s++){
    
    arma::mat diff_matrix = arma::zeros(num_pol+1,2);
    diff_matrix.col(0).fill(btc_diff);
    diff_matrix.col(1).fill(bch_diff);
    
    arma::mat running_matrix = arma::zeros(num_pol+1,2);
    running_matrix.col(0).fill(btc_difficulty_count);
    running_matrix.col(1).fill(bch_difficulty_count);
    
    arma::mat diff_change_matrix = arma::zeros(num_pol+1,2);
    diff_change_matrix.col(0).fill(btc_difficulty_change);
    diff_change_matrix.col(1).fill(bch_difficulty_change);
    
    arma::mat ts_current_matrix = arma::zeros(num_pol+1,2);
    ts_current_matrix.col(0).fill(interval);
    ts_current_matrix.col(1).fill(interval);
    
    arma::mat lagged_matrix = arma::zeros(num_pol+1,2);
    lagged_matrix.col(0).fill(btc_lagged);
    lagged_matrix.col(1).fill(bch_lagged);
    
    arma::mat lagged_other_matrix = arma::zeros(num_pol+1,2);
    lagged_other_matrix.col(0).fill(btc_lagged_other);
    lagged_other_matrix.col(1).fill(bch_lagged_other);
    
    arma::mat cumulative_matrix = arma::zeros(num_pol+1,12);
    cumulative_matrix.cols(0,5).each_row() = arma::conv_to<arma::rowvec>::from(btc_cumul);
    cumulative_matrix.cols(6,11).each_row() = arma::conv_to<arma::rowvec>::from(bch_cumul);
    
    NumericMatrix error_i_nm(T,2);
    error_i_nm = rmvnorm(T,mean_vec,correlation);
    arma::mat error_i(error_i_nm.begin(),T,2,false);
    
    NumericMatrix error_other_nm(T*5,2);
    error_other_nm = rmvnorm(T*5,mean_vec,correlation);
    arma::mat error_other(error_other_nm.begin(),T*5,2,false);
    
    List bch_eda_queue(num_pol+1);
    
    for(int i=0;i<=num_pol;i++){
      
      bch_eda_queue[i]=clone(as<List>(queue_list[queue_index-1]));
    }
    
    for( int t=0;t<T;t++){
      //cout<<"t: "<<t<<endl;
      //print(bch_eda_queue[0]);
      arma::mat ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                        s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),lagged_matrix(0,0),lagged_matrix(0,1),lagged_other_matrix(0,0),lagged_other_matrix(0,1));
      arma::rowvec x1 = ex_mt.row(0);
      arma::rowvec x2 = ex_mt.row(1);
      //cout<<"Exog"<<endl;
      //ex_mt.print();
      arma::vec decisions_actual_i = policy_function(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                       ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_i(t,0),error_i(t,1));
      //cout<<"Decisions by i"<<endl;
      //decisions_actual_i.print();
      
      arma::mat other_firms_decisions = other_firms(s_firm,x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                      ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_other(arma::span(5*t,5*(t+1)-1),arma::span::all));
      
      //cout<<"Decisions by other"<<endl;
      //other_firms_decisions.print();
      
      results(0)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute(decisions_actual_i,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),t,lagged_matrix(0,0),lagged_matrix(0,1),
              cumulative_matrix(arma::span(0),arma::span(0,5)),cumulative_matrix(arma::span(0),arma::span(6,11)),index_cumul,compute_col);
      
      //cout<<"profits vector"<<endl;
      //results(0)(span(s),span(t),span::all).print();
      
      total_btc_act=decisions_actual_i(0)+arma::sum(other_firms_decisions.col(0));
      total_bch_act=decisions_actual_i(1)+arma::sum(other_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(0,0)=decisions_actual_i(0);
      lagged_other_matrix(0,0)=sum(other_firms_decisions.col(0));
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),eda_vector(t), q_item_temp,mechanism);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism);
        
      }
      //cout<<"bch"<<endl;
      //cout<<total_bch_act<<endl;
      //cout<<"bch_diff"<<endl;
      //cout<<diff_matrix(0,1)<<endl;
      
      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)<1503428400){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)>=1503428400){
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(0,1)=decisions_actual_i(1);
      lagged_other_matrix(0,1)=sum(other_firms_decisions.col(1));
      
      ts_current_matrix.row(0)+=1800;
      
      row_ind<<0;
      
      cumulative_matrix(0,s_firm-1)=cumulative_matrix(0,s_firm-1)+decisions_actual_i(0);
      cumulative_matrix(0,s_firm-1+6)=cumulative_matrix(0,s_firm-1+6)+decisions_actual_i(1);
      
      cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
      cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
      
    }
    
    
  }
  
  return results;
  
}


// [[Rcpp::export]]
arma::field<arma::cube> simulation_run_uniformerror(int s_firm, int num_pol, int compute_col,int interval, SEXP price_array, SEXP fee_array, SEXP firm_vec, int btc_difficulty_count,
                                       int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                       int btc_lagged, int bch_lagged,int btc_lagged_other, int bch_lagged_other, SEXP eda_vec, SEXP btc_cum, SEXP bch_cum, SEXP index_cum,
                                       int S, int T, SEXP actual_policy_sexp,SEXP alternate_policies_list_sexp, SEXP queue_list_sexp, int queue_index){
  
  
  arma::cube prices_array = as<arma::cube>(price_array);
  arma::cube fees_array = as<arma::cube>(fee_array);
  arma::vec firm_vector = as<arma::vec>(firm_vec);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_cumul = as<arma::vec>(btc_cum);
  arma::vec bch_cumul = as<arma::vec>(bch_cum);
  arma::uvec index_cumul = as<arma::uvec>(index_cum);
  List actual_policy(actual_policy_sexp);
  List alternate_policies_list(alternate_policies_list_sexp);
  List queue_list(queue_list_sexp);

  arma::field<arma::cube> results(num_pol+1);
  
  for(int i =0;i<=num_pol;i++){
    arma::cube inst=arma::zeros(S,T,compute_col);
    results(i)=inst;
  }
  
  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  arma::uvec row_ind(1);
  int mechanism;
  
  for(int s=0;s<S;s++){
    
    arma::mat diff_matrix = arma::zeros(num_pol+1,2);
    diff_matrix.col(0).fill(btc_diff);
    diff_matrix.col(1).fill(bch_diff);
    
    arma::mat running_matrix = arma::zeros(num_pol+1,2);
    running_matrix.col(0).fill(btc_difficulty_count);
    running_matrix.col(1).fill(bch_difficulty_count);
    
    arma::mat diff_change_matrix = arma::zeros(num_pol+1,2);
    diff_change_matrix.col(0).fill(btc_difficulty_change);
    diff_change_matrix.col(1).fill(bch_difficulty_change);
    
    arma::mat ts_current_matrix = arma::zeros(num_pol+1,2);
    ts_current_matrix.col(0).fill(interval);
    ts_current_matrix.col(1).fill(interval);
    
    arma::mat lagged_matrix = arma::zeros(num_pol+1,2);
    lagged_matrix.col(0).fill(btc_lagged);
    lagged_matrix.col(1).fill(bch_lagged);
    
    arma::mat lagged_other_matrix = arma::zeros(num_pol+1,2);
    lagged_other_matrix.col(0).fill(btc_lagged_other);
    lagged_other_matrix.col(1).fill(bch_lagged_other);
    
    arma::mat cumulative_matrix = arma::zeros(num_pol+1,12);
    cumulative_matrix.cols(0,5).each_row() = arma::conv_to<arma::rowvec>::from(btc_cumul);
    cumulative_matrix.cols(6,11).each_row() = arma::conv_to<arma::rowvec>::from(bch_cumul);
    
    NumericVector error_i_nm = Rcpp::runif(T*2);
    arma::mat error_i(error_i_nm.begin(),T,2,false);
    
    NumericVector error_other_nm = Rcpp::runif(T*10);
    arma::mat error_other(error_other_nm.begin(),T*5,2,false);
    
    List bch_eda_queue(num_pol+1);
    
    for(int i=0;i<=num_pol;i++){
      
      bch_eda_queue[i]=clone(as<List>(queue_list[queue_index-1]));
    }
    
    for( int t=0;t<T;t++){
      //cout<<"t: "<<t<<endl;
      //print(bch_eda_queue[0]);
      arma::mat ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                        s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),lagged_matrix(0,0),lagged_matrix(0,1),lagged_other_matrix(0,0),lagged_other_matrix(0,1));
      arma::rowvec x1 = ex_mt.row(0);
      arma::rowvec x2 = ex_mt.row(1);
      //cout<<"Exog"<<endl;
      //ex_mt.print();
      arma::vec decisions_actual_i = policy_function_uniformerror(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                       ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),0,error_i(t,0),error_i(t,1));
      //cout<<"Decisions by i"<<endl;
      //decisions_actual_i.print();
      
      arma::mat other_firms_decisions = other_firms_uniformerror(s_firm,x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                      ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),0,error_other(arma::span(5*t,5*(t+1)-1),arma::span::all));
      
      //cout<<"Decisions by other"<<endl;
      //other_firms_decisions.print();
      
      results(0)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute(decisions_actual_i,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),t,lagged_matrix(0,0),lagged_matrix(0,1),
              cumulative_matrix(arma::span(0),arma::span(0,5)),cumulative_matrix(arma::span(0),arma::span(6,11)),index_cumul,compute_col);
      
      //cout<<"profits vector"<<endl;
      //results(0)(span(s),span(t),span::all).print();
      
      total_btc_act=decisions_actual_i(0)+arma::sum(other_firms_decisions.col(0));
      total_bch_act=decisions_actual_i(1)+arma::sum(other_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(0,0)=decisions_actual_i(0);
      lagged_other_matrix(0,0)=sum(other_firms_decisions.col(0));
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),eda_vector(t), q_item_temp,mechanism);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism);
        
      }
      //cout<<"bch"<<endl;
      //cout<<total_bch_act<<endl;
      //cout<<"bch_diff"<<endl;
      //cout<<diff_matrix(0,1)<<endl;
      
      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)<1503428400){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)>=1503428400){
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(0,1)=decisions_actual_i(1);
      lagged_other_matrix(0,1)=sum(other_firms_decisions.col(1));
      
      ts_current_matrix.row(0)+=1800;
      
      row_ind<<0;
      
      cumulative_matrix(0,s_firm-1)=cumulative_matrix(0,s_firm-1)+decisions_actual_i(0);
      cumulative_matrix(0,s_firm-1+6)=cumulative_matrix(0,s_firm-1+6)+decisions_actual_i(1);
      
      cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
      cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
      
      for(int p =0;p<num_pol;p++){
        
        List new_policy(alternate_policies_list[p]);
        
        arma::mat ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                          s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                                          lagged_other_matrix(p+1,0),lagged_other_matrix(p+1,1));
        arma::rowvec x1 = ex_mt.row(0);
        arma::rowvec x2 = ex_mt.row(1);
        
        arma::vec decisions_modified = policy_function_uniformerror(x1,x2,as<arma::vec>(new_policy[0]),as<arma::vec>(new_policy[1])
                                                         ,as<arma::vec>(new_policy[2]),as<arma::vec>(new_policy[3]),0,error_i(t,0),error_i(t,1));
        
        results(p+1)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute(decisions_modified,error_i.row(t),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
                arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),t,lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                cumulative_matrix(arma::span(p+1),arma::span(0,5)),cumulative_matrix(arma::span(p+1),arma::span(6,11)),index_cumul,compute_col);
        
        total_btc_alt=decisions_modified(0)+arma::sum(other_firms_decisions.col(0));
        total_bch_alt=decisions_modified(1)+arma::sum(other_firms_decisions.col(1));
        
        new_btc_diff = btc_diff_adjustment(diff_matrix(p+1,0), running_matrix(p+1,0), total_btc_alt, diff_change_matrix(p+1,0), ts_current_matrix(p+1,0));
        
        if(new_btc_diff!=diff_matrix(p+1,0)){
          diff_change_matrix(p+1,0)=ts_current_matrix(p+1,0);
          diff_matrix(p+1,0)=new_btc_diff;
        }
        
        running_matrix(p+1,0)=((int)running_matrix(p+1,0)+total_btc_alt)%2016;
        lagged_matrix(p+1,0)=decisions_modified(0);
        lagged_other_matrix(p+1,0)=sum(other_firms_decisions.col(0));
        
        temp_bch = diff_matrix(p+1,1);
        mechanism=0;
        
        if(eda_vector(t)==1 && total_bch_alt>0 && total_bch_alt<15){
          
          for(int m=1;m<=total_bch_alt;m++){
            
            q_item_temp=bch_eda_queue[p+1];
            q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(p+1,1),eda_vector(t),temp_bch);
            new_diff=bch_diff_adjustment(temp_bch, running_matrix(p+1,1), m, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1),eda_vector(t), q_item_temp,mechanism);
            
            if(new_diff>temp_bch){
              temp_bch=new_diff;
              break;
            }
            
            temp_bch=new_diff;
          }
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff = temp_bch;
          
        } else {
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff=bch_diff_adjustment(diff_matrix(p+1,1),running_matrix(p+1,1), total_bch_alt, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1), eda_vector(t), bch_eda_queue[p+1],mechanism);
          
        }
        
        if(new_bch_diff == diff_matrix(p+1,1)){
          
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism!=0){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)<1503428400){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = total_bch_alt;
        } 
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)>=1503428400){
          
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        lagged_matrix(p+1,1)=decisions_modified(1);
        lagged_other_matrix(p+1,1)=sum(other_firms_decisions.col(1));
        
        ts_current_matrix.row(p+1)+=1800;
        
        row_ind<<p+1;
        
        cumulative_matrix(p+1,s_firm-1)=cumulative_matrix(p+1,s_firm-1)+decisions_modified(0);
        cumulative_matrix(p+1,s_firm-1+6)=cumulative_matrix(p+1,s_firm-1+6)+decisions_modified(1);
        
        cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
        cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
        
      }
      
    }
    
    
  }
  
  return results;
  
}

// [[Rcpp::export]]
arma::field<arma::cube> simulation_run_singleerror(int s_firm, int num_pol, int compute_col,int interval, SEXP price_array, SEXP fee_array, SEXP firm_vec, int btc_difficulty_count,
                                                    int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                                    int btc_lagged, int bch_lagged,int btc_lagged_other, int bch_lagged_other, SEXP eda_vec, SEXP btc_cum, SEXP bch_cum, SEXP index_cum,
                                                    int S, int T, SEXP actual_policy_sexp,SEXP alternate_policies_list_sexp, SEXP queue_list_sexp, int queue_index){
  
  
  arma::cube prices_array = as<arma::cube>(price_array);
  arma::cube fees_array = as<arma::cube>(fee_array);
  arma::vec firm_vector = as<arma::vec>(firm_vec);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_cumul = as<arma::vec>(btc_cum);
  arma::vec bch_cumul = as<arma::vec>(bch_cum);
  arma::uvec index_cumul = as<arma::uvec>(index_cum);
  List actual_policy(actual_policy_sexp);
  List alternate_policies_list(alternate_policies_list_sexp);
  List queue_list(queue_list_sexp);
  
  arma::field<arma::cube> results(num_pol+1);
  
  for(int i =0;i<=num_pol;i++){
    arma::cube inst=arma::zeros(S,T,compute_col);
    results(i)=inst;
  }
  
  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  arma::uvec row_ind(1);
  int mechanism;
  
  for(int s=0;s<S;s++){
    
    arma::mat diff_matrix = arma::zeros(num_pol+1,2);
    diff_matrix.col(0).fill(btc_diff);
    diff_matrix.col(1).fill(bch_diff);
    
    arma::mat running_matrix = arma::zeros(num_pol+1,2);
    running_matrix.col(0).fill(btc_difficulty_count);
    running_matrix.col(1).fill(bch_difficulty_count);
    
    arma::mat diff_change_matrix = arma::zeros(num_pol+1,2);
    diff_change_matrix.col(0).fill(btc_difficulty_change);
    diff_change_matrix.col(1).fill(bch_difficulty_change);
    
    arma::mat ts_current_matrix = arma::zeros(num_pol+1,2);
    ts_current_matrix.col(0).fill(interval);
    ts_current_matrix.col(1).fill(interval);
    
    arma::mat lagged_matrix = arma::zeros(num_pol+1,2);
    lagged_matrix.col(0).fill(btc_lagged);
    lagged_matrix.col(1).fill(bch_lagged);
    
    arma::mat lagged_other_matrix = arma::zeros(num_pol+1,2);
    lagged_other_matrix.col(0).fill(btc_lagged_other);
    lagged_other_matrix.col(1).fill(bch_lagged_other);
    
    arma::mat cumulative_matrix = arma::zeros(num_pol+1,12);
    cumulative_matrix.cols(0,5).each_row() = arma::conv_to<arma::rowvec>::from(btc_cumul);
    cumulative_matrix.cols(6,11).each_row() = arma::conv_to<arma::rowvec>::from(bch_cumul);
    
    NumericVector error_i_nm = Rcpp::runif(T,1e-16,1-1e-16);
    arma::mat error_i(T,2,arma::fill::zeros);
    error_i.each_col() = as<arma::vec>(error_i_nm);
    
    NumericVector error_other_nm = Rcpp::runif(T*5,1e-16,1-1e-16);
    arma::mat error_other(T*5,2,arma::fill::zeros);
    error_other.each_col() = as<arma::vec>(error_other_nm);
    
    List bch_eda_queue(num_pol+1);
    
    for(int i=0;i<=num_pol;i++){
      
      bch_eda_queue[i]=clone(as<List>(queue_list[queue_index-1]));
    }
    
    for( int t=0;t<T;t++){
      //cout<<"t: "<<t<<endl;
      //print(bch_eda_queue[0]);
      arma::mat ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                        s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),lagged_matrix(0,0),lagged_matrix(0,1),lagged_other_matrix(0,0),lagged_other_matrix(0,1));
      arma::rowvec x1 = ex_mt.row(0);
      arma::rowvec x2 = ex_mt.row(1);
      //cout<<"Exog"<<endl;
      //ex_mt.print();
      arma::vec decisions_actual_i = policy_function_uniformerror(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                                    ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),0,error_i(t,0),error_i(t,1));
      //cout<<"Decisions by i"<<endl;
      //decisions_actual_i.print();
      
      arma::mat other_firms_decisions = other_firms_uniformerror(s_firm,x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                                   ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),0,error_other(arma::span(5*t,5*(t+1)-1),arma::span::all));
      
      //cout<<"Decisions by other"<<endl;
      //other_firms_decisions.print();
      
      results(0)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute_singleerror(decisions_actual_i,error_i(t,0),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(0,0),diff_matrix(0,1),t,lagged_matrix(0,0),lagged_matrix(0,1),
              cumulative_matrix(arma::span(0),arma::span(0,5)),cumulative_matrix(arma::span(0),arma::span(6,11)),index_cumul,compute_col);
      
      //cout<<"profits vector"<<endl;
      //results(0)(span(s),span(t),span::all).print();
      
      total_btc_act=decisions_actual_i(0)+arma::sum(other_firms_decisions.col(0));
      total_bch_act=decisions_actual_i(1)+arma::sum(other_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(0,0)=decisions_actual_i(0);
      lagged_other_matrix(0,0)=sum(other_firms_decisions.col(0));
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),eda_vector(t), q_item_temp,mechanism);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism);
        
      }
      //cout<<"bch"<<endl;
      //cout<<total_bch_act<<endl;
      //cout<<"bch_diff"<<endl;
      //cout<<diff_matrix(0,1)<<endl;
      
      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)<1503428400){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)>=1503428400){
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(0,1)=decisions_actual_i(1);
      lagged_other_matrix(0,1)=sum(other_firms_decisions.col(1));
      
      ts_current_matrix.row(0)+=1800;
      
      row_ind<<0;
      
      cumulative_matrix(0,s_firm-1)=cumulative_matrix(0,s_firm-1)+decisions_actual_i(0);
      cumulative_matrix(0,s_firm-1+6)=cumulative_matrix(0,s_firm-1+6)+decisions_actual_i(1);
      
      cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
      cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
      
      for(int p =0;p<num_pol;p++){
        
        List new_policy(alternate_policies_list[p]);
        
        arma::mat ex_mt=exogenous_creator(arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),
                                          s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                                          lagged_other_matrix(p+1,0),lagged_other_matrix(p+1,1));
        arma::rowvec x1 = ex_mt.row(0);
        arma::rowvec x2 = ex_mt.row(1);
        
        arma::vec decisions_modified = policy_function_uniformerror(x1,x2,as<arma::vec>(new_policy[0]),as<arma::vec>(new_policy[1])
                                                                      ,as<arma::vec>(new_policy[2]),as<arma::vec>(new_policy[3]),0,error_i(t,0),error_i(t,1));
        
        results(p+1)(arma::span(s),arma::span(t),arma::span::all) = profits_precompute_singleerror(decisions_modified,error_i(t,0),other_firms_decisions,arma::vectorise(prices_array(arma::span(s),arma::span(t),arma::span::all)),
                arma::vectorise(fees_array(arma::span(s),arma::span(t),arma::span::all)),s_firm,eda_vector(t),diff_matrix(p+1,0),diff_matrix(p+1,1),t,lagged_matrix(p+1,0),lagged_matrix(p+1,1),
                cumulative_matrix(arma::span(p+1),arma::span(0,5)),cumulative_matrix(arma::span(p+1),arma::span(6,11)),index_cumul,compute_col);
        
        total_btc_alt=decisions_modified(0)+arma::sum(other_firms_decisions.col(0));
        total_bch_alt=decisions_modified(1)+arma::sum(other_firms_decisions.col(1));
        
        new_btc_diff = btc_diff_adjustment(diff_matrix(p+1,0), running_matrix(p+1,0), total_btc_alt, diff_change_matrix(p+1,0), ts_current_matrix(p+1,0));
        
        if(new_btc_diff!=diff_matrix(p+1,0)){
          diff_change_matrix(p+1,0)=ts_current_matrix(p+1,0);
          diff_matrix(p+1,0)=new_btc_diff;
        }
        
        running_matrix(p+1,0)=((int)running_matrix(p+1,0)+total_btc_alt)%2016;
        lagged_matrix(p+1,0)=decisions_modified(0);
        lagged_other_matrix(p+1,0)=sum(other_firms_decisions.col(0));
        
        temp_bch = diff_matrix(p+1,1);
        mechanism=0;
        
        if(eda_vector(t)==1 && total_bch_alt>0 && total_bch_alt<15){
          
          for(int m=1;m<=total_bch_alt;m++){
            
            q_item_temp=bch_eda_queue[p+1];
            q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(p+1,1),eda_vector(t),temp_bch);
            new_diff=bch_diff_adjustment(temp_bch, running_matrix(p+1,1), m, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1),eda_vector(t), q_item_temp,mechanism);
            
            if(new_diff>temp_bch){
              temp_bch=new_diff;
              break;
            }
            
            temp_bch=new_diff;
          }
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff = temp_bch;
          
        } else {
          
          bch_eda_queue[p+1]=bch_queue_update(bch_eda_queue[p+1],total_bch_alt,ts_current_matrix(p+1,1),eda_vector(t),diff_matrix(p+1,1));
          new_bch_diff=bch_diff_adjustment(diff_matrix(p+1,1),running_matrix(p+1,1), total_bch_alt, diff_change_matrix(p+1,1), ts_current_matrix(p+1,1), eda_vector(t), bch_eda_queue[p+1],mechanism);
          
        }
        
        if(new_bch_diff == diff_matrix(p+1,1)){
          
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism!=0){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)<1503428400){
          
          diff_change_matrix(p+1,1)=ts_current_matrix(p+1,1);
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = total_bch_alt;
        } 
        
        if(new_bch_diff != diff_matrix(p+1,1) && mechanism==0 && ts_current_matrix(p+1,1)>=1503428400){
          
          diff_matrix(p+1,1)=new_bch_diff;
          running_matrix(p+1,1) = ((int)running_matrix(p+1,1)+total_bch_alt)%2016;
        }
        
        lagged_matrix(p+1,1)=decisions_modified(1);
        lagged_other_matrix(p+1,1)=sum(other_firms_decisions.col(1));
        
        ts_current_matrix.row(p+1)+=1800;
        
        row_ind<<p+1;
        
        cumulative_matrix(p+1,s_firm-1)=cumulative_matrix(p+1,s_firm-1)+decisions_modified(0);
        cumulative_matrix(p+1,s_firm-1+6)=cumulative_matrix(p+1,s_firm-1+6)+decisions_modified(1);
        
        cumulative_matrix(row_ind,index_cumul)=cumulative_matrix(row_ind,index_cumul)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(0));
        cumulative_matrix(row_ind,index_cumul+6)=cumulative_matrix(row_ind,index_cumul+6)+arma::conv_to<arma::rowvec>::from(other_firms_decisions.col(1));
        
      }
      
    }
    
    
  }
  
  return results;
  
}

// use counterfactual_simulation_mod for running simulations
// [[Rcpp::export]]
arma::cube counterfactual_simulation(int compute_col,int interval, SEXP price_array, SEXP fee_array, int btc_difficulty_count,
                                       int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                       SEXP btc_lagged_sexp, SEXP bch_lagged_sexp,SEXP btc_lagged_other_sexp, SEXP bch_lagged_other_sexp, SEXP eda_vec,
                                       int S, int T, SEXP actual_policy_sexp, SEXP queue_list_sexp, int queue_index){
  
  
  arma::mat prices_array = as<arma::mat>(price_array);
  arma::mat fees_array = as<arma::mat>(fee_array);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_lagged = as<arma::vec>(btc_lagged_sexp);
  arma::vec bch_lagged = as<arma::vec>(bch_lagged_sexp);
  arma::vec btc_lagged_other = as<arma::vec>(btc_lagged_other_sexp);
  arma::vec bch_lagged_other = as<arma::vec>(bch_lagged_other_sexp);
  
  Function rmvnorm = Environment("package:mvtnorm")["rmvnorm"];
  List actual_policy(actual_policy_sexp);
  List queue_list(queue_list_sexp);
  
  NumericMatrix correlation(2,2);
  correlation(0,0)=correlation(1,1)=1.0;
  correlation(0,1)=correlation(1,0)=std::tanh(as<double>(actual_policy[4]));
  NumericVector mean_vec=NumericVector::create(0,0);
  arma::cube results=arma::zeros(S,T,compute_col);

  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  int mechanism;
  int s_firm=0;
  arma::vec lagged_vec = arma::zeros(6);
  
  for(int s=0;s<S;s++){
    
    arma::mat diff_matrix = arma::zeros(1,2);
    diff_matrix(0,0)=btc_diff;
    diff_matrix(0,1)=bch_diff;
    
    arma::mat running_matrix = arma::zeros(1,2);
    running_matrix(0,0)=btc_difficulty_count;
    running_matrix(0,1)=bch_difficulty_count;
    
    arma::mat diff_change_matrix = arma::zeros(1,2);
    diff_change_matrix(0,0)=btc_difficulty_change;
    diff_change_matrix(0,1)=bch_difficulty_change;
    
    arma::mat ts_current_matrix = arma::zeros(1,2);
    ts_current_matrix(0,0)=interval;
    ts_current_matrix(0,1)=interval;
    
    arma::mat lagged_matrix = arma::zeros(1,12);
    lagged_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(btc_lagged);
    lagged_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(bch_lagged);
    
    arma::mat lagged_other_matrix = arma::zeros(1,12);
    lagged_other_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(btc_lagged_other);
    lagged_other_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(bch_lagged_other);
    
    NumericMatrix error_other_nm(T*6,2);
    error_other_nm = rmvnorm(T*6,mean_vec,correlation);
    arma::mat error_other(error_other_nm.begin(),T*6,2,false);
    
    List bch_eda_queue(1);
    bch_eda_queue[0]=clone(as<List>(queue_list[queue_index-1]));

    for( int t=0;t<T;t++){

      arma::mat ex_mt=counterfactual_exogenous_creator(arma::vectorise(prices_array(arma::span(t),arma::span::all)),
                                                       arma::vectorise(fees_array(arma::span(t),arma::span::all)),
                                       eda_vector(t),diff_matrix(0,0),diff_matrix(0,1));
      arma::rowvec x1 = ex_mt.row(0);
      arma::rowvec x2 = ex_mt.row(1);

      arma::mat all_firms_decisions = firms_decisions(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                      ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_other(arma::span(6*t,6*(t+1)-1),arma::span::all),
                                                      lagged_matrix,lagged_other_matrix);

      results(arma::span(s),arma::span(t),arma::span::all) = counterfactual_profits(all_firms_decisions,arma::vectorise(prices_array(arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(t),arma::span::all)),diff_matrix(0,0),diff_matrix(0,1));

      total_btc_act=arma::sum(all_firms_decisions.col(0));
      total_bch_act=arma::sum(all_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(all_firms_decisions.col(0));
      for(int i=0;i<6;i++){
        lagged_vec(i)=total_btc_act - all_firms_decisions(i,0);
      }
      lagged_other_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(lagged_vec);
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),eda_vector(t), q_item_temp,mechanism);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism);
        
      }

      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)<1503428400){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && ts_current_matrix(0,1)>=1503428400){
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(all_firms_decisions.col(1));
      for(int i=0;i<6;i++){
        lagged_vec(i)=total_bch_act - all_firms_decisions(i,1);
      }
      lagged_other_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(lagged_vec);
      
      ts_current_matrix.row(0)+=1800;
      
    }
    
    
  }
  
  return results;
  
}

double bch_diff_adjustment_mod(const double& current_diff, const int& running_coins, const int& coins_current, const int& ts_last, 
                           const int& ts_current, const int& eda,const List& bch_queue, int& mechanism, int& timediff, double& redfactor){
  
  if(coins_current==0){
    return current_diff;
  }
  
  int ind1,ind2;
  double ts_diff;
  List bch_queue_temp=clone(bch_queue);
  
  arma::vec q1 = arma::reverse(as<arma::vec>(bch_queue_temp[0])); //time intervals
  arma::vec q2 = arma::reverse(as<arma::vec>(bch_queue_temp[1])); // bch coins
  arma::uvec ps1,ps2;
  
  if(eda==1){
    
    ps1=arma::find(arma::cumsum(q2)>=6);
    ps2=arma::find(arma::cumsum(q2)>=12);
    
    if(ps1.is_empty() || ps2.is_empty()){
      return current_diff;
    }
    
    ind1 = ps1(0);
    ind2 = ps2(0);
    
    ts_diff=(double)q1(ind1)-q1(ind2);
    
    if(ts_diff>timediff){
      
      mechanism=0;
      return current_diff + std::log(redfactor);
      
    } else if(running_coins+coins_current>=2016){
      
      ts_diff=(double)ts_current-ts_last;
      
      if(ts_diff<14*86400*0.25){
        ts_diff=14*86400*0.25;
      }
      
      if(ts_diff>14*86400*4){
        ts_diff=14*86400*4;
      }
      
      mechanism=1;
      return current_diff-std::log((ts_diff/(14*86400)));
      
    } else {
      
      return current_diff;
    }
  } else {
    
    arma::vec q3 = arma::reverse(as<arma::vec>(bch_queue_temp[2])); // bch difficulty
    double workdone=0.0;
    ps1=arma::find(arma::cumsum(q2)>=2);
    ps2=arma::find(arma::cumsum(q2)>=145);
    
    if(ps1.is_empty() || ps2.is_empty()){
      return current_diff;
    }
    
    ind1 = ps1(0);
    ind2 = ps2(0);
    
    ts_diff=(double)q1(ind1)-q1(ind2);
    workdone=arma::mean(q3(arma::span(ind1,ind2)));
    
    if(ts_diff<86400*0.5){
      ts_diff=86400*0.5;
    }
    if(ts_diff>86400*2){
      ts_diff=86400*2;
    }
    
    mechanism=2;
    return current_diff-std::log((ts_diff/(86400)))-current_diff+workdone;
  }
  
}


// [[Rcpp::export]]
arma::cube counterfactual_simulation_mod(int compute_col,int interval, SEXP price_array, SEXP fee_array, int btc_difficulty_count,
                                     int btc_difficulty_change, int bch_difficulty_count, int bch_difficulty_change, double btc_diff, double bch_diff,
                                     SEXP btc_lagged_sexp, SEXP bch_lagged_sexp,SEXP btc_lagged_other_sexp, SEXP bch_lagged_other_sexp, SEXP eda_vec,
                                     int S, int T, SEXP actual_policy_sexp, SEXP queue_list_sexp, int queue_index,
                                     int timediff, double redfactor){
  
  
  arma::mat prices_array = as<arma::mat>(price_array);
  arma::mat fees_array = as<arma::mat>(fee_array);
  arma::vec eda_vector = as<arma::vec>(eda_vec);
  arma::vec btc_lagged = as<arma::vec>(btc_lagged_sexp);
  arma::vec bch_lagged = as<arma::vec>(bch_lagged_sexp);
  arma::vec btc_lagged_other = as<arma::vec>(btc_lagged_other_sexp);
  arma::vec bch_lagged_other = as<arma::vec>(bch_lagged_other_sexp);
  int first_adjust=0;
  
  Function rmvnorm = Environment("package:mvtnorm")["rmvnorm"];
  List actual_policy(actual_policy_sexp);
  List queue_list(queue_list_sexp);
  
  NumericMatrix correlation(2,2);
  correlation(0,0)=correlation(1,1)=1.0;
  correlation(0,1)=correlation(1,0)=std::tanh(as<double>(actual_policy[4]));
  NumericVector mean_vec=NumericVector::create(0,0);
  arma::cube results=arma::zeros(S,T,compute_col);
  
  int total_btc_act, total_bch_act,total_btc_alt, total_bch_alt;
  double new_btc_diff, new_bch_diff, temp_bch, new_diff;
  List q_item_temp;
  int mechanism;
  int s_firm=0;
  arma::vec lagged_vec = arma::zeros(6);
  
  for(int s=0;s<S;s++){
    
    arma::mat diff_matrix = arma::zeros(1,2);
    diff_matrix(0,0)=btc_diff;
    diff_matrix(0,1)=bch_diff;
    
    arma::mat running_matrix = arma::zeros(1,2);
    running_matrix(0,0)=btc_difficulty_count;
    running_matrix(0,1)=bch_difficulty_count;
    
    arma::mat diff_change_matrix = arma::zeros(1,2);
    diff_change_matrix(0,0)=btc_difficulty_change;
    diff_change_matrix(0,1)=bch_difficulty_change;
    
    arma::mat ts_current_matrix = arma::zeros(1,2);
    ts_current_matrix(0,0)=interval;
    ts_current_matrix(0,1)=interval;
    
    arma::mat lagged_matrix = arma::zeros(1,12);
    lagged_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(btc_lagged);
    lagged_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(bch_lagged);
    
    arma::mat lagged_other_matrix = arma::zeros(1,12);
    lagged_other_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(btc_lagged_other);
    lagged_other_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(bch_lagged_other);
    
    NumericMatrix error_other_nm(T*6,2);
    error_other_nm = rmvnorm(T*6,mean_vec,correlation);
    arma::mat error_other(error_other_nm.begin(),T*6,2,false);
    
    List bch_eda_queue(1);
    bch_eda_queue[0]=clone(as<List>(queue_list[queue_index-1]));
    
    for( int t=0;t<T;t++){
      
      arma::mat ex_mt=counterfactual_exogenous_creator(arma::vectorise(prices_array(arma::span(t),arma::span::all)),
                                                       arma::vectorise(fees_array(arma::span(t),arma::span::all)),
                                                       eda_vector(t),diff_matrix(0,0),diff_matrix(0,1));
      arma::rowvec x1 = ex_mt.row(0);
      arma::rowvec x2 = ex_mt.row(1);
      
      arma::mat all_firms_decisions = firms_decisions(x1,x2,as<arma::vec>(actual_policy[0]),as<arma::vec>(actual_policy[1])
                                                        ,as<arma::vec>(actual_policy[2]),as<arma::vec>(actual_policy[3]),actual_policy[5],error_other(arma::span(6*t,6*(t+1)-1),arma::span::all),
                                                        lagged_matrix,lagged_other_matrix);
      
      results(arma::span(s),arma::span(t),arma::span::all) = counterfactual_profits(all_firms_decisions,arma::vectorise(prices_array(arma::span(t),arma::span::all)),
              arma::vectorise(fees_array(arma::span(t),arma::span::all)),diff_matrix(0,0),diff_matrix(0,1));
      
      total_btc_act=arma::sum(all_firms_decisions.col(0));
      total_bch_act=arma::sum(all_firms_decisions.col(1));
      
      new_btc_diff = btc_diff_adjustment(diff_matrix(0,0), running_matrix(0,0), total_btc_act, diff_change_matrix(0,0), ts_current_matrix(0,0));
      
      if(new_btc_diff!=diff_matrix(0,0)){
        diff_change_matrix(0,0)=ts_current_matrix(0,0);
        diff_matrix(0,0)=new_btc_diff;
      }
      
      running_matrix(0,0)=((int)running_matrix(0,0)+total_btc_act)%2016;
      lagged_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(all_firms_decisions.col(0));
      for(int i=0;i<6;i++){
        lagged_vec(i)=total_btc_act - all_firms_decisions(i,0);
      }
      lagged_other_matrix(arma::span::all,arma::span(0,5)) = arma::conv_to<arma::rowvec>::from(lagged_vec);
      
      temp_bch = diff_matrix(0,1);
      mechanism=0;
      
      if(eda_vector(t)==1 && total_bch_act>0 && total_bch_act<15){
        
        for(int m=1;m<=total_bch_act;m++){
          
          q_item_temp=bch_eda_queue[0];
          q_item_temp=bch_queue_update(q_item_temp,m,ts_current_matrix(0,1),eda_vector(t),temp_bch);
          new_diff=bch_diff_adjustment_mod(temp_bch, running_matrix(0,1), m, diff_change_matrix(0,1), ts_current_matrix(0,1),
                                       eda_vector(t), q_item_temp,mechanism,timediff, redfactor);
          
          if(new_diff>temp_bch){
            temp_bch=new_diff;
            break;
          }
          
          temp_bch=new_diff;
        }
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff = temp_bch;
        
      } else {
        
        bch_eda_queue[0]=bch_queue_update(bch_eda_queue[0],total_bch_act,ts_current_matrix(0,1),eda_vector(t),diff_matrix(0,1));
        new_bch_diff=bch_diff_adjustment_mod(diff_matrix(0,1),running_matrix(0,1), total_bch_act, diff_change_matrix(0,1), ts_current_matrix(0,1), 
                                         eda_vector(t), bch_eda_queue[0],mechanism,timediff, redfactor);
        
      }
      
      if(mechanism==1 && first_adjust==0){
        first_adjust=1;
      }
      
      if(new_bch_diff == diff_matrix(0,1)){
        
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism!=0){
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && first_adjust==0){ //ts_current_matrix(0,1)<1503428400
        
        diff_change_matrix(0,1)=ts_current_matrix(0,1);
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = total_bch_act;
      }
      
      if(new_bch_diff != diff_matrix(0,1) && mechanism==0 && first_adjust==1){ //ts_current_matrix(0,1)>=1503428400
        
        diff_matrix(0,1)=new_bch_diff;
        running_matrix(0,1) = ((int)running_matrix(0,1)+total_bch_act)%2016;
      }
      
      lagged_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(all_firms_decisions.col(1));
      for(int i=0;i<6;i++){
        lagged_vec(i)=total_bch_act - all_firms_decisions(i,1);
      }
      lagged_other_matrix(arma::span::all,arma::span(6,11)) = arma::conv_to<arma::rowvec>::from(lagged_vec);
      
      ts_current_matrix.row(0)+=1800;
      
    }
    
    
  }
  
  return results;
  
}
