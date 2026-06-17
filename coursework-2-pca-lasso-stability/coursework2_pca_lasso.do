* ECM323: Big Data Analytics Coursework Assessment 2 Do File

clear all
set more off

cd "/Users/elleherzog/Downloads/MSc IBE & S/Big Data/Assessment 2/"
capture mkdir "figures"
capture mkdir "tables"


*QUESTION 1

*1a dataset 1
use "PCA_main_1.dta", clear
local xvars x1-x30

pca `xvars'

* saving component variances
matrix V1 = e(Ev)'
preserve
    clear
    svmat double V1, name(v)
    gen comp = _n
    gen prop = v1 / 30
    gen cum  = sum(prop)
    rename (v1 prop cum) (var_d1 prop_d1 cum_d1)
    save "tables/_var_d1.dta", replace
restore

* Scree plot
screeplot, yline(1, lpattern(dash) lcolor(red))                              ///
    title("Elbow plot - Dataset 1") ytitle("Variance per component")         ///
    name(scree1, replace)
graph export "figures/q1_scree_d1.png", replace as(png) width(1400)

*1a dataset 2
use "PCA_main_2.dta", clear
local xvars x1-x30
pca `xvars'

matrix V2 = e(Ev)'
preserve
    clear
    svmat double V2, name(v)
    gen comp = _n
    gen prop = v1 / 30
    gen cum  = sum(prop)
    rename (v1 prop cum) (var_d2 prop_d2 cum_d2)
    save "tables/_var_d2.dta", replace
restore

screeplot, yline(1, lpattern(dash) lcolor(red))                              ///
    title("Elbow plot - Dataset 2") ytitle("Variance per component")         ///
    name(scree2, replace)
graph export "figures/q1_scree_d2.png", replace as(png) width(1400)

* combined table of variances
preserve
    use "tables/_var_d1.dta", clear
    merge 1:1 comp using "tables/_var_d2.dta", nogen
    order comp var_d1 prop_d1 cum_d1 var_d2 prop_d2 cum_d2
    list comp var_d1 prop_d1 cum_d1 var_d2 prop_d2 cum_d2 in 1/10
    export delimited using "tables/pca_table.csv", replace
restore

* cumulative variance plot with both datasets
preserve
    use "tables/_var_d1.dta", clear
    merge 1:1 comp using "tables/_var_d2.dta", nogen
    twoway (line cum_d1 comp, lcolor(navy))                                  ///
           (line cum_d2 comp, lcolor(cranberry)),                            ///
        yline(0.80, lpattern(dot) lcolor(gs6))                               ///
        legend(order(1 "Dataset 1" 2 "Dataset 2") cols(2) ring(0) pos(5))    ///
        ytitle("Cumulative proportion of variance")                          ///
        xtitle("Number of components")                                       ///
        name(cum_both, replace)
    graph export "figures/q1_cumvar.png", replace as(png) width(1400)
restore


*QUESTION 2
* Implements Algorithm 12.1 from ISL §12.5.2 (PCA-based matrix completion).
* The data has 10% of cells in x1-x30 missing at random. The idea is to fill
* the missing cells with their best low-rank PCA prediction, refit PCA on
* the completed matrix, and repeat until the observed-cell MSE settles.

use "PCA_main_1_missing.dta", clear
unab xvars : x1-x30

* STEP 1 - standardise each variable using its OBSERVED-cell mean and SD,
* so every column enters PCA on the same scale. The flag miss_`v' records
* which cells were missing, since those cells need to be tracked across
* iterations. Missing cells are initialised at 0 in standardised units
* (the column mean), which is the recommended ISL starting value.
foreach v of local xvars {
    gen byte miss_`v' = missing(`v')
    quietly summarize `v'
    scalar m_`v' = r(mean)
    scalar s_`v' = r(sd)
    gen double z`v' = (`v' - m_`v') / s_`v'
    replace  z`v' = 0 if miss_`v' == 1
}

* STEP 2 - build a single list `zlist' of the 30 standardised columns so it
* can be passed to pca in one line.
local zlist
foreach v of local xvars {
    local zlist `zlist' z`v'
}

* STEP 3 - iterate Algorithm 12.1. At each pass:
*   (i)   fit a rank-M=5 PCA on the current Z (M carried over from Q1),
*   (ii)  use predict, fit to get the rank-5 reconstruction zhat,
*   (iii) compute the MSE between Z and zhat on the OBSERVED cells only,
*   (iv)  check convergence by the relative MSE drop, and
*   (v)   if not converged, overwrite the MISSING cells in Z with zhat
*         (observed cells are left untouched).
* This is the EM-style update from the textbook: the loss weakly decreases
* every step and stops when the observed-cell fit stops improving.

local mss0   = .       /* first-iteration MSE; used to normalise rel_err */
local mssold = .       /* previous-iteration MSE */

forvalues iter = 1/200 {

    * (i) fit the rank-5 PCA on the current standardised matrix
    quietly pca `zlist', components(5) covariance

    * (ii) extract the rank-5 fitted values for every cell. predict, fit
    * returns 30 new variables zhat1-zhat30, one per input column.
    capture drop zhat*
    quietly predict double zhat*, fit

    * (iii) accumulate the squared error on observed cells only, summed
    * across all 30 variables, then divide by the number of observed cells.
    local tot_sq  = 0
    local tot_obs = 0
    local i = 0
    foreach v of local xvars {
        local ++i
        quietly count if miss_`v' == 0
        local tot_obs = `tot_obs' + r(N)
        quietly gen double err = (z`v' - zhat`i')^2 if miss_`v' == 0
        quietly summarize err
        local tot_sq = `tot_sq' + r(sum)
        drop err
    }
    local mss = `tot_sq' / `tot_obs'

    * (iv) convergence check. Save iter-1 MSE as the baseline mss0; from
    * iter 2 onwards stop when the relative drop falls below 10^-7.
    if `iter' == 1 {
        local mss0   = `mss'
        local mssold = `mss'
    }
    else {
        local rel = (`mssold' - `mss') / `mss0'
        display "Iter " %2.0f `iter' "  mss = " %9.4f `mss' ///
                "  rel_err = " %12.4e `rel'
        if `rel' < 1e-7 {
            display "converged at iter `iter'."
            continue, break
        }
        local mssold = `mss'
    }

    * (v) overwrite ONLY the missing cells in Z with the rank-5 fit.
    * Observed cells stay anchored so the loss is always defined on the
    * same set of points.
    local i = 0
    foreach v of local xvars {
        local ++i
        quietly replace z`v' = zhat`i' if miss_`v' == 1
    }
}
capture drop zhat*

* STEP 4 - de-standardise the imputed cells back to the original units of
* x1-x30 (multiply by SD, add mean). Observed cells are left alone.
foreach v of local xvars {
    replace `v' = z`v' * s_`v' + m_`v' if miss_`v' == 1
}

*2b
count if miss_x16 == 1
local n16 = r(N)

twoway (scatter x16 myx16 if miss_x16 == 1,                                  ///
            msize(vsmall) mcolor(navy%40))                                   ///
       (lfit x16 myx16 if miss_x16 == 1, lcolor(forest_green) lwidth(medium)) ///
       (function y = x, range(-4 4) lpattern(dash) lcolor(red)),             ///
       legend(order(2 "OLS fit" 3 "45 degrees") cols(2) ring(0) pos(5))      ///
       aspectratio(1)                                                        ///
       ytitle("Matrix-completed x16") xtitle("True value myx16")             ///
       title("Assigned vs true x16 -- `n16' originally-missing cases")       ///
       name(q2_scatter, replace)
graph export "figures/q2_x16_scatter.png", replace as(png) width(1400)

* r and RMSE between completed x and true my x
tempfile acc
postfile pf str8 variable int n_miss double pearson_r rmse using "`acc'", replace
foreach v of local xvars {
    quietly {
        count if miss_`v' == 1
        local nm = r(N)
        correlate `v' my`v' if miss_`v' == 1
        local r = r(rho)
        gen double sq = (`v' - my`v')^2 if miss_`v' == 1
        summarize sq, meanonly
        local rmse = sqrt(r(mean))
        drop sq
    }
    post pf ("`v'") (`nm') (`r') (`rmse')
}
postclose pf
preserve
    use "`acc'", clear
    list, noobs sep(0) abbrev(12)
restore

drop z* miss_*


*QUESTION 3

use "housing_main.dta", clear

unab allvars : _all
local exclude ___KEY_SAMPLE_VARS__ lambdasample lassofolds                   ///
              ___HOUSING_VARS__ houseprice
local predictors : list allvars - exclude
display as text "Number of predictors: " as result `:word count `predictors''

*3a
preserve
    keep if lambdasample == 1
    lasso linear houseprice `predictors',                                    ///
        selection(cv, folds(10)) rseed(12345)
    scalar lam_star = e(lambda_sel)
    display as text "Optimal lambda* = " as result %12.4f lam_star
    cvplot, name(cvplot, replace)
    graph export "figures/q3_cv_curve.png", replace as(png) width(1400)
restore

*3b LASSO at lam_star on each of the 10 lassofolds.
tempfile fold_results
postfile pf int fold str40 predictor double coef using "`fold_results'", replace

forvalues k = 1/10 {
    preserve
        keep if lassofolds == `k'
        quietly count
        local nk = r(N)

        lasso linear houseprice `predictors',                                ///
            selection(none) rseed(12345)
        lassoselect lambda = `=scalar(lam_star)'

        local sel `e(allvars_sel)'
        local nsel : word count `sel'

        matrix bmat = e(b)

        foreach v of local sel {
            local c = bmat[1, "`v'"]
            post pf (`k') ("`v'") (`c')
        }
        display as text "Fold `k': N=`nk', selected " `nsel' " predictors"
    restore
}
postclose pf

preserve
    use "`fold_results'", clear
    bysort predictor: gen freq = _N
    duplicates drop predictor, force
    keep predictor freq
    gsort -freq
    save "tables/q3_selection_freq.dta", replace
    list predictor freq, noobs sep(0)
restore

preserve
    use "`fold_results'", clear
    gen byte selected = 1
    keep fold predictor selected
    reshape wide selected, i(predictor) j(fold)
    forvalues f = 1/10 {
        replace selected`f' = 0 if missing(selected`f')
    }
    local sumJ = 0
    local nPairs = 0
    forvalues i = 1/9 {
        local j1 = `i' + 1
        forvalues j = `j1'/10 {
            count if selected`i' == 1 & selected`j' == 1
            local inter = r(N)
            count if selected`i' == 1 | selected`j' == 1
            local uni = r(N)
            if `uni' > 0 {
                local sumJ = `sumJ' + `inter'/`uni'
            }
            local nPairs = `nPairs' + 1
        }
    }
    scalar MEAN_JACC = `sumJ' / `nPairs'
    display as text "Mean pairwise Jaccard across folds = "                  ///
           as result %5.3f scalar(MEAN_JACC)
restore

* visualisation 1: bar chart of selection frequency
preserve
    use "tables/q3_selection_freq.dta", clear
    keep if freq > 0
    gsort -freq
    graph bar (mean) freq, over(predictor, sort(freq) descending             ///
        label(angle(90) labsize(vsmall)))                                    ///
        ytitle("Selected in N of 10 folds") yscale(range(0 10))              ///
        title("Selection frequency at lambda* (only predictors selected >=1 time)")
    graph export "figures/q3_selection_freq_bar.png", replace as(png) width(2000)
restore

* visualisation 2: heatmap of predictors (y) x folds (x)
preserve
    use "`fold_results'", clear
    gen byte selected = 1
    keep fold predictor selected
    * attach selection frequency
    bysort predictor: egen freq = total(selected)
    gsort -freq predictor
    egen rank = group(predictor)
    sort rank fold

    twoway (scatter rank fold,                                               ///
                msymbol(square) mcolor(navy) msize(small)),                  ///
           legend(off) ylabel(none) xlabel(1(1)10) xtitle("Fold")            ///
           ytitle("Predictors (ranked by selection frequency)")              ///
           title("LASSO variable selection across 10 folds")                 ///
           name(heat, replace)
    graph export "figures/q3_selection_heatmap.png", replace as(png) width(1400)
restore

* coefficient-stability
preserve
    use "`fold_results'", clear
    bysort predictor: gen freq = _N
    keep if freq == 10
    bysort predictor: egen mean_coef = mean(coef)
    bysort predictor: egen sd_coef   = sd(coef)
    duplicates drop predictor, force
    gen cv_coef  = sd_coef / abs(mean_coef)
    gen abs_mean = abs(mean_coef)
    gsort -abs_mean
    format mean_coef sd_coef %10.0f
    format cv_coef           %5.2f
    list predictor mean_coef sd_coef cv_coef, noobs sep(0)
restore

