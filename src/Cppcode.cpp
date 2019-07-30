# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

const float PII = 3.14159265358979323846;


// [[Rcpp::export]]
float mcp_cpp(float t, float lambda, float gamma){
	if(fabs(t) < (lambda * gamma)){
    return (2*lambda * gamma * fabs(t) - pow(t,2)) / (2 * gamma);
	} else {
		return pow(lambda,2) * gamma / 2;
	}
}

// [[Rcpp::export]]
float scad_cpp(float t, float lambda, float gamma){
  if(fabs(t) < (lambda)){
    return lambda * fabs(t);
  } else{
    if(fabs(t) < gamma * lambda){
      return (gamma * lambda * fabs(t) - 0.5 * (pow(t,2) + pow(lambda,2))) / (gamma - 1);
    } else{
      return pow(lambda,2) * (pow(gamma,2) -1) / (2 * (gamma - 1));
    }
  }
}

// [[Rcpp::export]]
float norm_quotient_cpp(arma::vec beta, float lambda){
  float n1 = norm(beta,1);
  if(n1 == 0){
    return 1;
  } else{
    return lambda * n1 / norm(beta,2);
  }
}

// [[Rcpp::export]]
float elastic_net_cpp(float t, float lambda, float gamma){
  return lambda * gamma * fabs(t) + lambda * ((1-gamma)/2) * pow(t,2);
}

// [[Rcpp::export]]
float norm_quotient_mod_cpp(arma::vec beta, float lambda, float constant){
  return lambda * norm(beta,1) / (norm(beta,2) + constant);
}

// [[Rcpp::export]]
float pnorm_cpp(float x)
{
  // constants
  float a1 =  0.254829592;
  float a2 = -0.284496736;
  float a3 =  1.421413741;
  float a4 = -1.453152027;
  float a5 =  1.061405429;
  float p  =  0.3275911;
  
  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x)/sqrt(2.0);
  
  // A&S formula 7.1.26
  float t = 1.0/(1.0 + p*x);
  float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
  
  return 0.5*(1.0 + sign*y);
}


// [[Rcpp::export]]
float aux_fun_1(float s){
  if(s > 100){
    return s;
  } else{
    return log(1 + exp(s));
  }
}

// [[Rcpp::export]]
float rho_ch_cpp(float t, float k = 0.5){
  if(t <= k){
    return t*exp(-sqrt(k));
  } else{
    return -2*exp(-sqrt(t))*(1+sqrt(t)) + exp(-sqrt(k)) * (2 * (1+sqrt(k)) + k);
  }
}

// [[Rcpp::export]]
float dev_cpp(float score, int y){
  float aux = aux_fun_1(score);
  return -y * (score - aux) + (1-y) * aux;
}

// [[Rcpp::export]]
float der_rho_ch_cpp(float t, float k = 0.5){
  if(t <= k){
    return exp(-sqrt(k));
  } else{
    return exp(-sqrt(t));
  }
}

// [[Rcpp::export]]
float G_ch_cpp(float t, float k = 0.5){
  if(t <= exp(-k)){
    return t*exp(-sqrt(-log(t))) + exp(0.25)*sqrt(PII) * (pnorm_cpp(sqrt(2)*(0.5 + sqrt(-log(t)))) - 1);
  } else{
    return exp(-sqrt(k))*t + exp(0.25)*sqrt(PII)*(pnorm_cpp(sqrt(2)*(0.5 + sqrt(k)))-1);
  }
}

// [[Rcpp::export]]
float inv_logit_cpp(float u){
  if(u > 16){
    return 1;
  }
  if(u < -16){
    return 0;
  }
  return 1/(1+exp(-u));
}

// [[Rcpp::export]]
float phi_ch_cpp(float score, int y){
  float prob = inv_logit_cpp(score);
  return rho_ch_cpp(dev_cpp(score,y)) + G_ch_cpp(prob) + G_ch_cpp(1-prob) - G_ch_cpp(1);
}



// [[Rcpp::export]]
float der_phi_0_ch_cpp(float score){
  float prob = inv_logit_cpp(score);
  float aux = aux_fun_1(score);
  return der_rho_ch_cpp(aux) * pow(prob,2) + der_rho_ch_cpp(-score + aux) * prob * (1 - prob);
}


// [[Rcpp::export]]
float der_phi_ch_cpp(float score, int y){
  if(y == 0){
    return der_phi_0_ch_cpp(score);
  } else{
    return -der_phi_0_ch_cpp(-score);
  }
}


// [[Rcpp::export]]
arma::vec get_scores(arma::mat X, arma::vec beta, float beta0){
  return(X*beta + beta0);
}


// [[Rcpp::export]]
float eval_loss_function_ch_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  arma::vec scores = get_scores(X,beta,beta0);
  float acum = 0;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * phi_ch_cpp(scores[i], Y[i]);
  }
  return acum/n;
}

// [[Rcpp::export]]
arma::vec eval_der_loss_function_ch_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec scores = get_scores(X,beta,beta0);
  arma::vec acum = zeros<vec>(p) ;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * der_phi_ch_cpp(scores[i], Y[i]) * X.row(i).t();
  }
  return acum/n;
}







// [[Rcpp::export]]
float der_dev_cpp(float score, int y){
  return inv_logit_cpp(score) - y;
}



// [[Rcpp::export]]
float eval_loss_function_dev_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  arma::vec scores = get_scores(X,beta,beta0);
  float acum = 0;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * dev_cpp(scores[i], Y[i]);
  }
  return acum/n;
}

// [[Rcpp::export]]
arma::vec eval_der_loss_function_dev_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec scores = get_scores(X,beta,beta0);
  arma::vec acum = zeros<vec>(p) ;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * der_dev_cpp(scores[i], Y[i]) * X.row(i).t();
  }
  return acum/n;
}










// [[Rcpp::export]]
float der_rho_div_cpp(float t, float c = 0.5){
  return (c + 1)*exp(-c*t);
}

// [[Rcpp::export]]
float G_div_cpp(float t, float c = 0.5){
  return pow(t,c+1);
}

// [[Rcpp::export]]
float rho_div_cpp(float t, float c = 0.5){
  return (1+1/c)*(1 - exp(-c*t));
}

// [[Rcpp::export]]
float phi_div_cpp(float score, int y){
  float prob = inv_logit_cpp(score);
  return rho_div_cpp(dev_cpp(score,y)) + G_div_cpp(prob) + G_div_cpp(1-prob) - G_div_cpp(1);
}


// [[Rcpp::export]]
float der_phi_0_div_cpp(float score, float c = 0.5){
  float prob = inv_logit_cpp(score);
  return prob * (c+1) * (pow(prob,c) * (1-prob) + pow(1-prob,c) * prob);
}


// [[Rcpp::export]]
float der_phi_div_cpp(float score, int y){
  if(y == 0){
    return der_phi_0_div_cpp(score);
  } else{
    return -der_phi_0_div_cpp(-score);
  }
}


// [[Rcpp::export]]
float eval_loss_function_div_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  arma::vec scores = get_scores(X,beta,beta0);
  float acum = 0;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * phi_div_cpp(scores[i], Y[i]);
  }
  return acum/n;
}

// [[Rcpp::export]]
arma::vec eval_der_loss_function_div_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec scores = get_scores(X,beta,beta0);
  arma::vec acum = zeros<vec>(p) ;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * der_phi_div_cpp(scores[i], Y[i]) * X.row(i).t();
  }
  return acum/n;
}




// [[Rcpp::export]]
float phi_lse_cpp(float score, int y){
  return pow(y - inv_logit_cpp(score),2);
}

// [[Rcpp::export]]
float der_phi_lse_cpp(float score, int y){
  float prob = inv_logit_cpp(score);
  return -2 * (y-prob) * prob * (1-prob);
}



// [[Rcpp::export]]
float eval_loss_function_lse_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  arma::vec scores = get_scores(X,beta,beta0);
  float acum = 0;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * phi_lse_cpp(scores[i], Y[i]);
  }
  return acum/n;
}

// [[Rcpp::export]]
arma::vec eval_der_loss_function_lse_cpp(arma::mat X, arma::vec Y, arma::vec beta, float beta0, arma::vec weights){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec scores = get_scores(X,beta,beta0);
  arma::vec acum = zeros<vec>(p) ;
  int i;
  for(i=0;i < n; i++){
    acum += weights[i] * der_phi_lse_cpp(scores[i], Y[i]) * X.row(i).t();
  }
  return acum/n;
}




