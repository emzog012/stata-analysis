ECM323: Big Data Analytics — Coursework Do File


  Required packages (run once):
    ssc install crossfold
    ssc install rforest

global data  "/Users/elleherzog/Downloads/MSc IBE & S/Big Data/Assessment 1/"
global out   "/Users/elleherzog/Downloads/MSc IBE & S/Big Data/Assessment 1/Output/"

ssc install crossfold, replace
ssc install rforest, replace

set more off
set seed 42


*QUESTION 1
  
use "$data/main_2025_26_data_1.dta", clear

summarize y e
summarize x1-x100

*1a

lasso linear y x1-x100, folds(10) selection(cv) rseed(42)
lassoinfo
lassocoef, display(coef, postselection) sort(coef, postselection)

lasso linear y x1-x100, folds(10) selection(cv, serule) rseed(42)
lassoinfo
lassocoef, display(coef, postselection) sort(coef, postselection)

*Graphs

lasso linear y x1-x100, folds(10) rseed(42)

coefpath
graph export "$out/q1_coefpath.png", replace width(2000)

lasso linear y x1-x100, folds(10) selection(cv, serule) rseed(42)

cvplot
graph export "$out/q1_cv_mse.png", replace width(2000)

*1b

regress y x14 x30 x49 x83, robust


*1c: 
regress y x14 x30 x49 x83
predict resid_1se, resid
correlate resid_1se e

foreach v in x6 x25 x29 x33 x38 x41 x45 x51 x70 x84 x85 {
    regress y x14 x30 x49 x83 `v'
    predict resid_ext, resid
    quietly correlate resid_ext e
    di "Adding `v': corr(resid, e) = " r(rho)
    drop resid_ext
}


*QUESTION 2

use "/Users/elleherzog/Downloads/MSc IBE & S/Big Data/main_2025_26_data_2.dta", clear

summarize houseprice distance

scatter houseprice distance, msize(vtiny) scheme(s1mono) ///
    title("House Price vs Distance from Tube Station") ///
    xtitle("Distance (metres)") ytitle("House Price (£)")
graph export "$out/q2_scatter.png", replace width(2000)

summarize distance
local dmean = r(mean)
local dsd   = r(sd)
gen dist_s = (distance - `dmean') / `dsd'

forvalues p = 1/10 {
    gen dist_s`p' = dist_s^`p'
}

* variable lists 
global plist0 ""
forvalues p = 1/10 {
    local pminus1 = `p' - 1
    global plist`p' "${plist`pminus1'} dist_s`p'"
}

*2a/2b

gen polynomial = .
gen cv_mse     = .
gen cv_se      = .

forvalues p = 1/10 {
    crossfold regress houseprice ${plist`p'}, k(10) loud
    matrix cv_folds = r(est)
    local sum   = 0
    local sumsq = 0
    forvalues k = 1/10 {
        local mse_k = cv_folds[`k',1]^2
        local sum   = `sum'   + `mse_k'
        local sumsq = `sumsq' + `mse_k'^2
    }
    local mean_mse = `sum' / 10
    local se_mse   = sqrt((`sumsq'/10 - `mean_mse'^2)) / sqrt(10)
    replace polynomial = `p'         in `p'
    replace cv_mse     = `mean_mse'  in `p'
    replace cv_se      = `se_mse'    in `p'
    di "p=`p': CV MSE = `mean_mse'  SE = `se_mse'"
}

*Plotting

twoway (connected cv_mse polynomial, lcolor(navy) mcolor(navy) msymbol(circle)), ///
    xlabel(1/10) xline(4, lcolor(red) lpattern(dash)) ///
    xtitle("Polynomial Order (p)") ytitle("10-Fold CV MSE") ///
    title("CV MSE by Polynomial Order") scheme(s1mono)
graph export "$out/q2_cv_mse.png", replace width(2000)

*2c

regress houseprice dist_s1 dist_s2 dist_s3 dist_s4, robust


*QUESTION 3

global X "co2_emissions_current number_habitable_rooms open_fire_dummy extension_dummy main_fuel_gas floorareaQ10 i.myageband i.myproperty_type"

*OLS

use "$data/train_main_2025_26.dta", clear

regress efficiency_gap $X, robust

* Training MSE
predict yhat_ols
gen sq_err_ols = (efficiency_gap - yhat_ols)^2
summarize sq_err_ols
di "OLS Train MSE: " r(mean)

estimates save "$out/ols_model", replace

* Test MSE
use "$data/test_main_2025_26.dta", clear
estimates use "$out/ols_model"
predict yhat_ols_test
gen sq_err_ols_test = (efficiency_gap - yhat_ols_test)^2
summarize sq_err_ols_test
di "OLS Test MSE: " r(mean)

*LASSO

use "$data/train_main_2025_26.dta", clear

lasso linear efficiency_gap $X, folds(10) selection(cv) rseed(42)
lassoinfo
lassocoef, display(coef, postselection) sort(coef, postselection)

* Training MSE
predict yhat_lasso
gen sq_err_lasso = (efficiency_gap - yhat_lasso)^2
summarize sq_err_lasso
di "LASSO Train MSE: " r(mean)

* Test MSE
use "$data/train_main_2025_26.dta", clear
gen train = 1
append using "$data/test_main_2025_26.dta"
replace train = 0 if train == .

lasso linear efficiency_gap $X if train == 1, folds(10) selection(cv) rseed(42)
predict yhat_lasso_all
gen sq_err_lasso_test = (efficiency_gap - yhat_lasso_all)^2 if train == 0
summarize sq_err_lasso_test if train == 0
di "LASSO Test MSE: " r(mean)

*RANDOM FOREST

use "$data/train_main_2025_26.dta", clear

global X_rf "co2_emissions_current number_habitable_rooms open_fire_dummy extension_dummy main_fuel_gas floorareaQ10 myageband myproperty_type"

forvalues m = 1/8 {
    rforest efficiency_gap $X_rf, type(reg) numvars(`m') iterations(50)
    di "m = `m'  OOB MSE = " e(OOB_Error)
}

* Fit final
local m_opt 4


rforest efficiency_gap $X_rf, type(reg) numvars(4) iterations(100)
di "RF OOB MSE: " e(OOB_Error)

* Variable importance
matrix vim = e(importance)
matrix list vim

* Training MSE
predict yhat_rf
gen sq_err_rf = (efficiency_gap - yhat_rf)^2
summarize sq_err_rf
di "RF Train MSE: " r(mean)

* Test MSE 
use "$data/train_main_2025_26.dta", clear
gen train = 1
append using "$data/test_main_2025_26.dta"
replace train = 0 if train == .

rforest efficiency_gap $X_rf if train == 1, ///
    type(reg) numvars(4) iterations(100)
predict yhat_rf_all

gen sq_err_rf_test = (efficiency_gap - yhat_rf_all)^2 if train == 0
summarize sq_err_rf_test if train == 0
di "RF Test MSE: " r(mean)
