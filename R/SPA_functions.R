
ScoreTest_wSaddleApprox_Get_X1 = function(X1)
{
	q1<-ncol(X1)
	if(q1>=2)
	{
		if(sum(abs(X1[,1]-X1[,2]))==0)
		{
			X1=X1[,-2]
			q1<-q1-1
		}
	}
	qr1<-qr(X1)
	if(qr1$rank < q1){
		
		X1.svd<-svd(X1)
		X1 = X1.svd$u[,1:qr1$rank]
	} 

	return(X1)
}


ScoreTest_wSaddleApprox_NULL_Model <- function(formula, data=NULL)
{
	X1<-model.matrix(formula,data=data)
	X1<-ScoreTest_wSaddleApprox_Get_X1(X1)
	
	glmfit= glm(formula, data=data, family = "binomial")
	convflag=0
	if(glmfit$converged)
	{
		mu = glmfit$fitted.values
		if(mean(mu)/mean(glmfit$y)>0.001 & (1-mean(mu))/(1-mean(glmfit$y))>0.001)	convflag<-1	#Check that the null model converged properly with glm
	}
	if(convflag==0)
	{
		firthfit=fast.logistf.fit(x=X1,y=glmfit$y)
		eta<-X1%*%firthfit$beta
  		mu    = as.vector(exp(eta)/(1+exp(eta)))
	}

	V = mu*(1-mu)
  	res = glmfit$y- mu
	n1<-length(res)
	
	XV = t(X1 * V)
	XVX_inv= solve(t(X1)%*%(X1 * V))
	XXVX_inv= X1 %*% XVX_inv   
	
	re<-list(y=glmfit$y, mu=mu, res=res, V=V, X1=X1, XV=XV, XXVX_inv =XXVX_inv)
	class(re)<-"SA_NULL"
	return(re)	
}

Korg<-function(t, mu, g)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp<-log(1 - mu + mu * exp(g* t1))
		out[i]<-sum(temp)
	}
	return(out)
}


K1_adj<-function(t, mu, g, q)
{
	n.t<-length(t)	
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp1<-(1 - mu)* exp(-g * t1) + mu
		temp2<-mu *g
		out[i]<-sum(temp2/temp1)-q
	}
	return(out)
}

K2<-function(t, mu, g)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp1<-((1 - mu)* exp(-g * t1) + mu)^2
		temp2<-(1-mu) * mu * g^2 * exp(-g*t1)
		out[i]<-sum(temp2/temp1,na.rm=TRUE)
	}
	return(out)
}

Get_Normal_Spline<-function(var,nodes)
{
	y1<-var*nodes
	y2<-rep(var,length(nodes))
	return(cbind(y1,y2))
}

Get_Saddle_Spline<-function(mu,g,nodes)
{
	m1<-sum(mu*g)
	y1<-K1_adj(nodes,mu,g,0)-m1
	y2<-K2(nodes,mu,g)
	return(cbind(nodes,y1,y2)[which(nodes!=0),])
}


getroot_K1<-function(init,mu,g,q,m1,tol=.Machine$double.eps^0.25,maxiter=1000)
{
	g.pos<-sum(g[which(g>0)])
	g.neg<- sum(g[which(g<0)])
	if(q>=g.pos || q<=g.neg)
	{
		return(list(root=Inf,n.iter=0,Is.converge=TRUE))
	} else {
		t<-init
		K1_eval<-K1_adj(t,mu,g,q)
		prevJump<- Inf
		rep<-1
		repeat
		{
			K2_eval<-K2(t,mu,g)
			tnew<-t-K1_eval/K2_eval
			if(is.na(tnew))
			{
				conv=FALSE
				break
			}
			if(abs(tnew-t)<tol)
			{
				conv<-TRUE
				break
			}
			if(rep==maxiter)
			{
				conv<-FALSE
				break
			}

			newK1<-K1_adj(tnew,mu,g,q)
			if(sign(K1_eval)!=sign(newK1))
			{
				if(abs(tnew-t)>prevJump-tol)
				{
					tnew<-t+sign(newK1-K1_eval)*prevJump/2
					newK1<-K1_adj(tnew,mu,g,q)
					prevJump<-prevJump/2
				} else {
					prevJump<-abs(tnew-t)
				}
			}

			rep<-rep+1
			t<-tnew
			K1_eval<-newK1
		}
		return(list(root=t,n.iter=rep,Is.converge=conv))
	}
}

Get_Saddle_Prob<-function(zeta, mu, g, q) 
{
	k1<-Korg(zeta, mu, g)
	k2<-K2(zeta, mu, g)
	
	if(is.finite(k1) && is.finite(k2))
	{
	temp1<-zeta * q - k1

	
	w<-sign(zeta) * (2 *temp1)^{1/2}
	v<- zeta * (k2)^{1/2}
			
	Z.test<-w + 1/w * log(v/w)	

	if(Z.test > 0){
		pval<-pnorm(Z.test, lower.tail = FALSE)
	} else {
		pval= -pnorm(Z.test, lower.tail = TRUE)
	}	
	} else {
	pval<-0
	}
	
	return(pval)
}
	
Saddle_Prob<-function(q, mu, g, Cutoff=2,alpha,output="P",nodes.fixed,nodes.init)
{
	m1<-sum(mu * g)
	var1<-sum(mu * (1-mu) * g^2)
	p1=NULL
	p2=NULL
	if(output=="metaspline")
	{
		if(is.null(nodes.fixed)==TRUE)
		{
			nodes<-nodes.init
			nodes<-getnodes(nodes,mu,g)$nodes
		} else {
			nodes<-unique(c(nodes.fixed,0))
			nodes<-nodes[order(nodes)]
		}
	}

	Score<- q-m1
	#
	qinv = -sign(q-m1) * abs(q-m1) + m1

	# Noadj
	pval.noadj<-pchisq((q - m1)^2/var1, lower.tail = FALSE, df=1)
	Is.converge=TRUE

 	if(Cutoff=="BE"){
		rho<-sum(((abs(g))^3)*mu*(1-mu)*(mu^2+(1-mu)^2))
		B<-0.56*rho*var1^(-3/2)
		p<-B+alpha/2
		Cutoff=ifelse(p>=0.496,0.01,qnorm(p,lower.tail=F))
	} else if(Cutoff < 10^-1){
		Cutoff=10^-1
	} 			
	#
	if(output=="metaspline")
	{
		splfun<-Get_Saddle_Spline(mu,g,nodes)
	}
	if(abs(q - m1)/sqrt(var1) < Cutoff ){

		pval=pval.noadj		
	} else {
		out.uni1<-getroot_K1(0, mu=mu, g=g, q=q)
		out.uni2<-getroot_K1(0, mu=mu, g=g, q=qinv)
		if(out.uni1$Is.converge==TRUE && out.uni2$Is.converge==TRUE)
		{
			p1<-Get_Saddle_Prob(out.uni1$root, mu, g, q)
			p2<-Get_Saddle_Prob(out.uni2$root, mu, g, qinv)
			pval = abs(p1)+abs(p2)
			Is.converge=TRUE
		} else {
 			print("Error_Converge")
			pval<-pval.noadj
			Is.converge=FALSE	
		}				
	}
	
	if(pval!=0 && pval.noadj/pval>10^3)
	{
		return(Saddle_Prob(q, mu, g, Cutoff=Cutoff*2,alpha,output,nodes.fixed,nodes.init))
	} else if(output=="metaspline")
	{
		return(list(p.value=pval, p.value.NA=pval.noadj, Is.converge=Is.converge,Score=Score,splfun=splfun,var=var1))
	} else {
		return(list(p.value=pval, p.value.NA=pval.noadj, Is.converge=Is.converge, Score=Score))
	}
}

TestSPA<-function(G, obj.null,  Cutoff=2,alpha,output,nodes.fixed=NULL,nodes.init)
{
	if(class(obj.null) != "SA_NULL"){
		stop("obj.null should be a returned object from ScoreTest_wSaddleApprox_NULL_Model")
	}
	
	y = obj.null$y
	mu = obj.null$mu
	res = obj.null$res
	
	n.g<-sum(G)
	if(n.g/(2*length(G))>0.5)
	{
		G<-2-G
		n.g<-sum(G)
	}
	G1<-G  -  obj.null$XXVX_inv %*%  (obj.null$XV %*% G)
	q<-sum(G1 * y)
	out<-Saddle_Prob(q, mu=mu, g=G1,  Cutoff=Cutoff,alpha=alpha,output=output,nodes.fixed=nodes.fixed,nodes.init=nodes.init)

	return(out)
}

Korg_fast<-function(t, mu, g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp<-log(1 - muNB + muNB * exp(gNB* t1))
		out[i]<-sum(temp)+NAmu*t1+0.5*NAsigma*t1^2
	}
	return(out)
}


K1_adj_fast<-function(t, mu, g, q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
{
	n.t<-length(t)	
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp1<-(1 - muNB)* exp(-gNB * t1) + muNB
		temp2<-muNB *gNB
		temp3<-NAmu+NAsigma*t1
		out[i]<-sum(temp2/temp1)+temp3-q
	}
	return(out)
}

K2_fast<-function(t, mu, g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp1<-((1 - muNB)* exp(-gNB * t1) + muNB)^2
		temp2<-(1-muNB) * muNB * gNB^2 * exp(-gNB*t1)
		out[i]<-sum(temp2/temp1,na.rm=TRUE)+NAsigma
	}
	return(out)
}

Get_Saddle_Spline_fast<-function(nodes,mu,g,gNA,gNB,muNA,muNB,NAmu,NAsigma,m1)
{
	y1<-K1_adj_fast(nodes,mu, g, 0,gNA,gNB,muNA,muNB,NAmu,NAsigma)-m1
	y2<-K2_fast(nodes,mu, g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
	return(cbind(nodes,y1,y2)[which(nodes!=0),])
}

getroot_K1_fast<-function(init,mu,g,q,m1,gNA,gNB,muNA,muNB,NAmu,NAsigma,tol=.Machine$double.eps^0.25,maxiter=1000)
{
	g.pos<-sum(g[which(g>0)])
	g.neg<- sum(g[which(g<0)])
	if(q>=g.pos || q<=g.neg)
	{
		return(list(root=Inf,n.iter=0,Is.converge=TRUE))
	} else {
		t<-init
		K1_eval<-K1_adj_fast(t,mu,g,q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
		prevJump<- Inf
		rep<-1
		repeat
		{
			K2_eval<-K2_fast(t,mu,g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
			tnew<-t-K1_eval/K2_eval
			if(is.na(tnew))
			{
				conv=FALSE
				break
			}
			if(abs(tnew-t)<tol)
			{
				conv<-TRUE
				break
			}
			if(rep==maxiter)
			{
				conv<-FALSE
				break
			}

			newK1<-K1_adj_fast(tnew,mu,g,q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
			if(sign(K1_eval)!=sign(newK1))
			{
				if(abs(tnew-t)>prevJump-tol)
				{
					tnew<-t+sign(newK1-K1_eval)*prevJump/2
					newK1<-K1_adj_fast(tnew,mu,g,q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
					prevJump<-prevJump/2
				} else {
					prevJump<-abs(tnew-t)
				}
			}

			rep<-rep+1
			t<-tnew
			K1_eval<-newK1
		}
		return(list(root=t,n.iter=rep,Is.converge=conv))
	}
}

Get_Saddle_Prob_fast<-function(zeta, mu, g, q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
{
	k1<-Korg_fast(zeta, mu, g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
	k2<-K2_fast(zeta, mu, g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
	
	if(is.finite(k1) && is.finite(k2))
	{
	temp1<-zeta * q - k1

	
	w<-sign(zeta) * (2 *temp1)^{1/2}
	v<- zeta * (k2)^{1/2}
			
	Z.test<-w + 1/w * log(v/w)	

	if(Z.test > 0){
		pval<-pnorm(Z.test, lower.tail = FALSE)
	} else {
		pval= -pnorm(Z.test, lower.tail = TRUE)
	}	
	} else {
	pval<-0
	}
	
	return(pval)
}

Saddle_Prob_fast<-function(q, g,mu,gNA,gNB,muNA,muNB,Cutoff=2,alpha,output,nodes.fixed,nodes.init)
{
	m1<-sum(mu * g)
	var1<-sum(mu * (1-mu) * g^2)
	p1=NULL
	p2=NULL
	NAmu= m1-sum(gNB*muNB)
	NAsigma=var1-sum(muNB*(1-muNB)*gNB^2)
	if(output=="metaspline")
	{
		if(is.null(nodes.fixed)==TRUE)
		{
			nodes<-nodes.init
			nodes<-getnodes_fast(nodes,mu,g,gNA,gNB,muNA,muNB,NAmu,NAsigma)$nodes
		} else {
			nodes<-unique(c(nodes.fixed,0))
			nodes<-nodes[order(nodes)]
		}
	}
	Score<- q-m1

	qinv = -sign(q-m1) * abs(q-m1) + m1

	# Noadj
	pval.noadj<-pchisq((q - m1)^2/var1, lower.tail = FALSE, df=1)
	Is.converge=TRUE

	if(Cutoff=="BE"){
		rho<-sum(((abs(g))^3)*mu*(1-mu)*(mu^2+(1-mu)^2))
		B<-0.56*rho*var1^(-3/2)
		p<-B+alpha/2
		Cutoff=ifelse(p>=0.496,0.01,qnorm(p,lower.tail=F))
	} else if(Cutoff < 10^-1){
		Cutoff=10^-1
	} 
			
	#	
	if(output=="metaspline")
	{
		splfun<-Get_Saddle_Spline_fast(nodes,mu,g,gNA,gNB,muNA,muNB,NAmu,NAsigma,m1)
	}
	if(abs(q - m1)/sqrt(var1) < Cutoff ){

		pval=pval.noadj
		
	} else {
		out.uni1<-getroot_K1_fast(0, mu=mu, g=g, q=q,gNA=gNA,gNB=gNB,muNA=muNA,muNB=muNB,NAmu=NAmu,NAsigma=NAsigma)
		out.uni2<-getroot_K1_fast(0, mu=mu, g=g, q=qinv,gNA=gNA,gNB=gNB,muNA=muNA,muNB=muNB,NAmu=NAmu,NAsigma=NAsigma)
		if(out.uni1$Is.converge==TRUE && out.uni2$Is.converge==TRUE)
		{
			p1<-Get_Saddle_Prob_fast(out.uni1$root, mu, g, q,gNA,gNB,muNA,muNB,NAmu,NAsigma)
			p2<-Get_Saddle_Prob_fast(out.uni2$root, mu, g, qinv,gNA,gNB,muNA,muNB,NAmu,NAsigma)
			pval = abs(p1)+abs(p2)
			Is.converge=TRUE
		} else {
 			print("Error_Converge")
			pval<-pval.noadj
			Is.converge=FALSE	
		}		
		
	}
	
	if(pval!=0 && pval.noadj/pval>10^3)
	{
		return(Saddle_Prob_fast(q, g,mu,gNA,gNB,muNA,muNB,Cutoff=Cutoff*2,alpha,output,nodes.fixed,nodes.init))
	} else if(output=="metaspline")
	{
		return(list(p.value=pval, p.value.NA=pval.noadj, Is.converge=Is.converge, Score=Score,splfun=splfun,var=var1))
	} else {
		return(list(p.value=pval, p.value.NA=pval.noadj, Is.converge=Is.converge, Score=Score))
	}
}

TestSPAfast<-function(G, obj.null, output, Cutoff=2,alpha,nodes.fixed,nodes.init)
{
	if(class(obj.null) != "SA_NULL"){
		stop("obj.null should be a returned object from ScoreTest_wSaddleApprox_NULL_Model")
	}
	
	y = obj.null$y
	mu = obj.null$mu
	res = obj.null$res
	
	n.g<-sum(G)
	if(n.g/(2*length(G))>0.5)
	{
		G<-2-G
		n.g<-sum(G)
	}
	NAset<-which(G==0)
	G1<-G  -  obj.null$XXVX_inv %*%  (obj.null$XV %*% G)
	q<-sum(G1 * y) 
	g=G1

	if(length(NAset)/length(G)<0.5)
	{
		out<-Saddle_Prob(q, mu=mu, g=g,  Cutoff=Cutoff,alpha=alpha, output=output,nodes.fixed=nodes.fixed,nodes.init=nodes.init)
	} else {

	out<-Saddle_Prob_fast(q,g=g,mu=mu,gNA=g[NAset],gNB=g[-NAset],
muNA=mu[NAset],muNB=mu[-NAset], Cutoff=Cutoff,alpha=alpha,output=output,nodes.fixed=nodes.fixed,nodes.init=nodes.init)
	}


	return(out)
}

fast.logistf.fit <- function (x, y, weight = NULL, offset = NULL, firth = TRUE, col.fit = NULL, 
    init = NULL, control) {
    n <- nrow(x)
    k <- ncol(x)
    if (is.null(init)) 
        init = rep(0, k)
    if (is.null(col.fit)) 
        col.fit = 1:k
    if (is.null(offset)) 
        offset = rep(0, n)
    if (is.null(weight)) 
        weight = rep(1, n)
    if (col.fit[1] == 0) 
        maxit <- 0
    if (missing(control)) 
        control <- fast.logistf.control()
    maxit <- control$maxit
    maxstep <- control$maxstep
    maxhs <- control$maxhs
    lconv <- control$lconv
    gconv <- control$gconv
    xconv <- control$xconv
    beta <- init
    iter <- 0
    pi <- as.vector(1/(1 + exp(-x %*% beta - offset)))
    evals <- 1
    repeat {
        beta.old <- beta
        XW2 <- t(x * (weight * pi * (1-pi))^0.5)
        myQR <- qr(t(XW2))
        Q <- qr.Q(myQR)
        h <- (Q*Q) %*% rep(1, ncol(Q))        
        if (firth) 
            U.star <- crossprod(x, weight * (y - pi) + h * (0.5 - pi))
        else U.star <- crossprod(x, weight * (y - pi))
        XX.covs <- matrix(0, k, k)
        if (col.fit[1] != 0) {
            XX.XW2 <- t(x[, col.fit, drop=FALSE] * (weight * pi * (1-pi))^0.5)
            XX.Fisher <- crossprod(t(XX.XW2))
            XX.covs[col.fit, col.fit] <- fast.invFisher(XX.Fisher)   ###### HERE IS THE PROBLEM!!!
        }
        if(all(is.na(XX.covs)) == T) {
            break
        }  
        delta <- as.vector(XX.covs %*% U.star)
        delta[is.na(delta)] <- 0
        mx <- max(abs(delta))/maxstep
        if (mx > 1) 
            delta <- delta/mx
        evals <- evals + 1
        if (maxit > 0) {
            iter <- iter + 1
            beta <- beta + delta
                pi <- as.vector(1/(1 + exp(-x %*% beta - offset)))

        }
        if (iter == maxit | ((max(abs(delta)) <= xconv) & (all(abs(U.star[col.fit]) < 
            gconv)))) 
            break
    }
    # Error catching (if chol(x) not positive definite)
    if(all(is.na(XX.covs))==T) {
        var <- XX.covs
        list(beta = NA, var = var, pi = NA, hat.diag = NA, 
  iter = NA, evals = NA, conv = c(NA, 
            NA, NA))
    } else {
        var <- XX.covs
        list(beta = beta, var = var, pi = pi, hat.diag = h, 
            iter = iter, evals = evals, conv = c(max(abs(U.star)),
		 max(abs(delta))))
    }
}


fast.logistf.control <- function (maxit = 50, maxhs = 15, maxstep = 15, lconv = 1e-05, 
    gconv = 1e-05, xconv = 1e-05) 
{
    list(maxit = maxit, maxhs = maxhs, maxstep = maxstep, lconv = lconv, 
        gconv = gconv, xconv = xconv)
}

fast.logDet <- function (x) {
    my.chol <- tryCatch(chol(x),error=function(e) {NA})
    if (all(is.na(my.chol))==T) {
        return(NA)
    } else {
        return (2 * sum(log(diag(my.chol))))
    }
}   

fast.invFisher <- function(x) {
  my.chol <- tryCatch(chol(x),error=function(e) {NA})
    if (all(is.na(my.chol))==T) {
        return(NA)
    } else {
        return (chol2inv(my.chol))
    }
  #ifelse(is.na(my.chol), NA, chol2inv(my.chol))
}

getnodes<-function(init,mu,g)
{
nodes<-unique(c(init,0))
nodes<-nodes[order(nodes)]
flag=0
rep<-0
t<-c(2^seq(-2,10,0.5),-2^seq(-2,10,0.5),0)
t<-t[order(t)]
yt<-K1_adj(t,mu,g,0)
w<-rep(1,length(t))
w[which(t^2>1)]=1/(abs(t[which(t^2>1)]))^(1/3)
jump<-0.5
totres<-Inf
repeat{
	y1<-K1_adj(nodes ,mu ,g,0)
	y2<-K2(nodes ,mu ,g)
	sfun<-splinefunH(nodes, y1 ,m=y2)
	pred<-sfun(t,deriv=0)
	res<-w*abs(pred-yt)
	newnode=NULL
	for(i in 1:length(nodes))
	{
		if(nodes[i]!=0)
		{
		if(i>1)
		{	
			int1<-which(t<nodes[i] & t>nodes[i-1])
		} else {
			int1<-which(t<nodes[i])
		}
		if(i<length(nodes))
		{
			int2<-which(t<nodes[i+1] & t>nodes[i])
		} else {
			int2<-which(t>nodes[i])
		}
		if(length(int1)>0)
		{
			r1=res[int1]
			t1=t[int1]
			t1<-t1[min(which(r1==max(r1)))]
			r1<-max(r1)
		} else {
			r1<-0
		}
		if(length(int2)>0)
		{
			r2=res[int2]
			t2=t[int2]
			t2<-t2[min(which(r2==max(r2)))]
			r2<-max(r2)
		} else {
			r2<-0
		}
		if(r1==r2)
		{
			newnode<-c(newnode,nodes[i])
		} else if(r1>r2)
		{
			jump1<- jump*(1-max(0.1,r2/r1))*abs(nodes[i])
			newnode<-c(newnode,nodes[i]-jump1)
		} else {
			jump1<- jump*(1-max(0.1,r1/r2))*abs(nodes[i])
			newnode<-c(newnode,nodes[i]+jump1)
		}
		}
	}
	newnode<-c(newnode,0)
		if(rep>100 || (sum(res)>totres)) break
		rep<-rep+1
	totres<-sum(res)
	nodes<-newnode[order(newnode)]

}
	return(list(nodes=nodes,loss=totres))
}


getnodes_fast<-function(init,mu,g,gNA,gNB,muNA,muNB,NAmu,NAsigma)
{
nodes<-unique(c(init,0))
nodes<-nodes[order(nodes)]
flag=0
rep<-0
t<-c(2^seq(-2,10,0.5),-2^seq(-2,10,0.5),0)
t<-t[order(t)]
yt<-K1_adj_fast(t,mu, g, 0,gNA,gNB,muNA,muNB,NAmu,NAsigma)
w<-rep(1,length(t))
w[which(t^2>1)]=1/(abs(t[which(t^2>1)]))^(1/3)
jump<-0.5
totres<-Inf
repeat{
	y1<-K1_adj(nodes ,mu ,g,0)
	y2<-K2(nodes ,mu ,g)
	sfun<-splinefunH(nodes, y1 ,m=y2)
	pred<-sfun(t,deriv=0)
	res<-w*abs(pred-yt)
	newnode=NULL
	for(i in 1:length(nodes))
	{
		if(nodes[i]!=0)
		{
		if(i>1)
		{	
			int1<-which(t<nodes[i] & t>nodes[i-1])
		} else {
			int1<-which(t<nodes[i])
		}
		if(i<length(nodes))
		{
			int2<-which(t<nodes[i+1] & t>nodes[i])
		} else {
			int2<-which(t>nodes[i])
		}
		if(length(int1)>0)
		{
			r1=res[int1]
			t1=t[int1]
			t1<-t1[min(which(r1==max(r1)))]
			r1<-max(r1)
		} else {
			r1<-0
		}
		if(length(int2)>0)
		{
			r2=res[int2]
			t2=t[int2]
			t2<-t2[min(which(r2==max(r2)))]
			r2<-max(r2)
		} else {
			r2<-0
		}
		if(r1==r2)
		{
			newnode<-c(newnode,nodes[i])
		} else if(r1>r2)
		{
			jump1<- jump*(1-max(0.1,r2/r1))*abs(nodes[i])
			newnode<-c(newnode,nodes[i]-jump1)
		} else {
			jump1<- jump*(1-max(0.1,r1/r2))*abs(nodes[i])
			newnode<-c(newnode,nodes[i]+jump1)
		}
		}
	}
	newnode<-c(newnode,0)
		if(rep>100 || (sum(res)>totres)) break
		rep<-rep+1
	totres<-sum(res)
	nodes<-newnode[order(newnode)]

}
	return(list(nodes=nodes,loss=totres))
}



ScoreTest_SPA <-function(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),minmac=5,Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7)
{
	method<-match.arg(method)

	if(missing(obj.null))
	{
		if(missing(cov) || is.null(cov))
		{
			cov<-rep(1,length(pheno))
		}
		obj.null<-ScoreTest_wSaddleApprox_NULL_Model(as.matrix(pheno) ~as.matrix(cov))
	}
	cov<-obj.null$X1
	pheno<-obj.null$y

	genos<-as.matrix(genos)
	if(ncol(genos)==1)
	{
		m<-1
		genos<-t(genos)
	} else {
		m <- nrow(genos)
	}
	if(is.na(missing.id)==FALSE)
	{
		genos[which(genos==missing.id)]<-NA
	}
	p.value<-rep(NA,m)
	p.value.NA<-rep(NA,m)
	Is.converge<-rep(NA,m)
	beta<-rep(NA,m)
	SEbeta<-rep(NA,m)
	for (i in 1:m)
	{
		try({
		ina<-which(is.na(genos[i,]))
		if(length(ina)>0)
		{
			genos[i,ina]<-mean(genos[i,],na.rm=TRUE)
		}
		MAC<-min(sum(genos[i,]),sum(2-genos[i,]))
		if(MAC>=minmac)
		{
			if(method=="fastSPA")
			{
				re <- TestSPAfast(as.vector(genos[i,,drop=FALSE]), obj.null, Cutoff=Cutoff, alpha=alpha, output="P")
			} else {
				re <- TestSPA(as.vector(genos[i,,drop=FALSE]), obj.null,Cutoff=Cutoff, alpha=alpha, output="P")
			}
			p.value[i] <- re$p.value
			p.value.NA[i] <- re$p.value.NA
			Is.converge[i]<- re$Is.converge
			if(beta.out==TRUE && p.value[i]<beta.Cutoff)
			{
				re.firth<-fast.logistf.fit(x=cbind(t(genos[i,,drop=FALSE]),cov),y=pheno,firth=TRUE)
				beta[i]<-re.firth$beta[1]
				SEbeta[i]<-sqrt(re.firth$var[1,1])
			}
		}
		})
		if(i%%1000==0)	print(paste("Processed",i,"SNPs",sep=" "))
	}
	return(list(p.value=p.value,p.value.NA=p.value.NA,Is.converge=Is.converge,beta=beta,SEbeta=SEbeta))
}


ScoreTest_SPA_wMeta <-function(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),minmac=5,Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7,output=c("P","metaZ","metaGC","metaspline"),nodes.fixed=NULL,nodes.init=c(-100,-10,-1,1,10,100))
{
	method<-match.arg(method)
	output<-match.arg(output)

	if(missing(obj.null))
	{
		obj.null<-ScoreTest_wSaddleApprox_NULL_Model(as.matrix(pheno) ~as.matrix(cov))
	}
	cov<-obj.null$X1
	pheno<-obj.null$y
	if(is.null(nodes.fixed))
	{
		nodes.init<-unique(nodes.init[which(nodes.init!=0)])
	} else {
		nodes.init<-unique(nodes.fixed[which(nodes.fixed!=0)])
	}

	genos<-as.matrix(genos)
	if(ncol(genos)==1)
	{
		m<-1
		genos<-t(genos)
	} else {
		m <- nrow(genos)
	}
	if(is.na(missing.id)==FALSE)
	{
		genos[which(genos==missing.id)]<-NA
	}
	p.value<-rep(NA,m)
	p.value.NA<-rep(NA,m)
	GCmat<-matrix(NA,m,2)
	Is.converge<-rep(NA,m)
	beta<-rep(NA,m)
	SEbeta<-rep(NA,m)
	Score<-rep(NA,m)
	#Skew<-rep(NA,m)
	nodes<-matrix(NA,m,length(nodes.init))
	splfuny1<-matrix(NA,m,length(nodes.init))
	splfuny2<-matrix(NA,m,length(nodes.init))
	Var<-rep(NA,m)
	MAF<-rep(NA,m)
	for (i in 1:m)
	{
		try({
		ina<-which(is.na(genos[i,]))
		if(length(ina)>0)
		{
			genos[i,ina]<-mean(genos[i,],na.rm=TRUE)
		}
		AC1<-sum(genos[i,])
		AC2<-sum(2-genos[i,])
		MAC<-min(AC1,AC2)
		if(MAC>=minmac)
		{
			if(method=="fastSPA")
			{
				re <- TestSPAfast(as.vector(genos[i,,drop=FALSE]), obj.null, Cutoff=Cutoff, alpha=alpha,output=output, nodes.fixed=nodes.fixed,nodes.init=nodes.init)
			} else {
				re <- TestSPA(as.vector(genos[i,,drop=FALSE]), obj.null,Cutoff=Cutoff, alpha=alpha,output=output, nodes.fixed=nodes.fixed,nodes.init=nodes.init)
			}
			p.value[i] <- re$p.value
			p.value.NA[i] <- re$p.value.NA
			Is.converge[i]<- re$Is.converge
			if(beta.out==TRUE && p.value[i]<beta.Cutoff)
			{
				re.firth<-fast.logistf.fit(x=cbind(t(genos[i,,drop=FALSE]),cov),y=pheno,firth=TRUE)
				beta[i]<-re.firth$beta[1]
				SEbeta[i]<-sqrt(re.firth$var[1,1])
			}
			if(output!="P")
			{
				GCmat[i,]<-c(length(which(round(genos[i,],0)==2)),length(which(round(genos[i,],0)==1)))
				if(AC1>AC2)	GCmat[i,]<-c(length(which(round(genos[i,],0)==0)),length(which(round(genos[i,],0)==1)))
				Score[i]<-re$Score
				MAF[i]<-MAC/(2*length(pheno))
				if(output=="metaspline")
				{
					nodes[i,1:nrow(re$splfun)]<-re$splfun[,1]
					splfuny1[i,1:nrow(re$splfun)]<-re$splfun[,2]
					splfuny2[i,1:nrow(re$splfun)]<-re$splfun[,3]
					Var[i]<-re$var
				}
			}
		}
		})
		if(i%%1000==0)	print(paste("Processed",i,"SNPs",sep=" "))
	}
	if(output=="P")
	{
		return(list(p.value=p.value,p.value.NA=p.value.NA,Is.converge=Is.converge,beta=beta,SEbeta=SEbeta))
	} else if(output=="metaZ"){
		return(list(p.value=sign(Score)*p.value,p.value.NA=sign(Score)*p.value.NA,Is.converge=Is.converge,beta=beta,SEbeta=SEbeta,MAF=MAF))
	} else if(output=="metaGC"){
		return(list(p.value=sign(Score)*p.value,p.value.NA=sign(Score)*p.value.NA,Is.converge=Is.converge,beta=beta,SEbeta=SEbeta,GCmat=GCmat))
	} else {
		return(list(p.value=sign(Score)*p.value,p.value.NA=sign(Score)*p.value.NA,Is.converge=Is.converge,beta=beta,SEbeta=SEbeta,GCmat=GCmat,spldata=cbind(Score,Var,nodes,splfuny1,splfuny2)))
	}
}

pval_MAC1<-function(q,p,m1=m1,var1=var1,GenC,gcent,truep,Cutoff=2)
{	
	n0<-GenC[1]
	n1<-GenC[2]
	n2<-GenC[3]
	re<-Saddle_Prob_MAC(q, p=p,m1=m1,var1=var1, n0=n0,n1=n1,n2=n2,g=gcent, Cutoff=Cutoff)
	return(re$p.value-truep)
}

Korg_MAC<-function(t, p, n0,n1,n2,g)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	for(i in 1:n.t){
		t1<-t[i]
		out[i]<-n1*log(1 - p + p * exp(g[2]*t1))+n2*log(1 - p + p * exp(g[3]*t1))+
		n0*log(1 - p + p * exp(g[1]*t1))
	}
	return(out)
}


K1_adj_MAC<-function(t, p, n0,n1,n2,g, q)
{
	n.t<-length(t)	
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp1<-n1*g[2]*p/((1 - p)* exp(-g[2]*t1) + p)+n2*g[3]*p/((1 - p)* exp(-g[3]*t1) + p)+
		n0*g[1]*p/((1 - p)* exp(-g[1]*t1) + p)
		out[i]<-temp1-q
	}
	return(out)
}

K2_MAC<-function(t, p,n0,n1,n2,g)
{
	n.t<-length(t)
	out<-rep(0,n.t)

	for(i in 1:n.t){
		t1<-t[i]
		temp<-rep(NA,3)
		temp[2]<-n1*(1-p) * g[2]^2* p * exp(-g[2]*t1)/(((1 - p)* exp(-g[2]*t1) + p)^2)
		temp[3]<-n2*(1-p) * g[3]^2* p * exp(-g[3]*t1)/(((1 - p)* exp(-g[3]*t1) + p)^2)
		temp[1]<-n0*(1-p) * g[1]^2* p * exp(-g[1]*t1)/(((1 - p)* exp(-g[1]*t1) + p)^2)
		out[i]<-sum(temp,na.rm=TRUE)
	}
	return(out)
}



getroot_K1_MAC<-function(init,p,n0,n1,n2,g,q,m1,tol=.Machine$double.eps^0.25,maxiter=1000)
{
	gtot<-c(n0,n1,n2)*g
	g.pos<-sum(gtot[which(gtot>0)])
	g.neg<- -g.pos
	if(q>=g.pos || q<=g.neg)
	{
		return(list(root=Inf,n.iter=0,Is.converge=TRUE))
	} else {
		t<-init
		K1_eval<-K1_adj_MAC(t,p,n0,n1,n2,g,q)
		prevJump<- Inf
		rep<-1
		repeat
		{
			K2_eval<-K2_MAC(t,p,n0,n1,n2,g)
			tnew<-t-K1_eval/K2_eval
			if(is.na(tnew))
			{
				conv=FALSE
				break
			}
			if(abs(tnew-t)<tol)
			{
				conv<-TRUE
				break
			}
			if(rep==maxiter)
			{
				conv<-FALSE
				break
			}

			newK1<-K1_adj_MAC(tnew,p,n0,n1,n2,g,q)
			if(sign(K1_eval)!=sign(newK1))
			{
				if(abs(tnew-t)>prevJump-tol)
				{
					tnew<-t+sign(newK1-K1_eval)*prevJump/2
					newK1<-K1_adj_MAC(tnew,p,n0,n1,n2,g,q)
					prevJump<-prevJump/2
				} else {
					prevJump<-abs(tnew-t)
				}
			}

			rep<-rep+1
			t<-tnew
			K1_eval<-newK1
		}
		return(list(root=t,n.iter=rep,Is.converge=conv))
	}
}

Get_Saddle_Prob_MAC<-function(zeta, p,n0,n1,n2,g, q) 
{
	k1<-Korg_MAC(zeta, p,n0,n1,n2,g)
	k2<-K2_MAC(zeta, p,n0,n1,n2,g)
	
	if(is.finite(k1) && is.finite(k2))
	{
	temp1<-zeta * q - k1

	
	w<-sign(zeta) * (2 *temp1)^{1/2}
	v<- zeta * (k2)^{1/2}
			
	Z.test<-w + 1/w * log(v/w)	

	if(Z.test > 0){
		pval<-pnorm(Z.test, lower.tail = FALSE)
	} else {
		pval= -pnorm(Z.test, lower.tail = TRUE)
	}	
	} else {
	pval<-0
	}
	
	return(pval)
}


Saddle_Prob_MAC<-function(q, p, m1,var1,n0,n1,n2,g, Cutoff=2)
{

	qinv = -sign(q-m1) * abs(q-m1) + m1

	# Noadj
	pval.noadj<-pchisq((q - m1)^2/var1, lower.tail = FALSE, df=1)
	p1<-sign(q-m1)*pval.noadj/2
	p2= -sign(q-m1)*pval.noadj/2
	Is.converge=TRUE

	if(Cutoff < 10^-2){
		Cutoff=10^-2
	} 			
	#
	if(abs(q - m1)/sqrt(var1) < Cutoff ){

		pval=pval.noadj		
	} else {
		out.uni1<-getroot_K1_MAC(0, p=p,n0=n0,n1=n1,n2=n2, g=g,q=q)
		out.uni2<-getroot_K1_MAC(0, p=p,n0=n0,n1=n1,n2=n2, g=g,q=qinv)
		if(out.uni1$Is.converge==TRUE && out.uni2$Is.converge==TRUE)
		{
			p1<-Get_Saddle_Prob_MAC(out.uni1$root, p,n0,n1,n2,g, q)
			p2<-Get_Saddle_Prob_MAC(out.uni2$root,  p,n0,n1,n2,g, qinv)
			pval = abs(p1)+abs(p2)
			Is.converge=TRUE
		} else {
 			print("Error_Converge")
			pval<-pval.noadj
			Is.converge=FALSE	
		}				
	}
	
	if(pval!=0 && pval.noadj/pval>10^3)
	{
		return(Saddle_Prob_MAC(q, p,m1,var1,n0,n1,n2,g, Cutoff=Cutoff*2))
	} else {
		return(list(p.value=pval, p.value.NA=pval.noadj, Is.converge=Is.converge, p1=p1, p2=p2))
	}
}



SPAmeta<-function(pvalue.Z=NULL,MAF.Z=NULL,CCsize.Z=NULL,
			pvalue.GC=NULL,GCmat=NULL,CCsize.GC=NULL,Cutoff.GC=2,
			spldata=NULL,
			Cutoff.meta=2)
{
	#Check all the inputs are correct
	info<-c(length(pvalue.Z),length(pvalue.GC))
	if(info[1]>0)
	{
		if(info[2]==0 & is.null(spldata))
		{
			return(MetaSPA_Normal(pvalue=pvalue.Z,MAF=MAF.Z,CCsize=CCsize.Z))
		} else {
			if(is.null(MAF.Z))	stop("MAF is needed when hybridizing Z-score based meta-analysis with other methods")
			if(length(which(is.na(MAF.Z)))>0 | length(MAF.Z)!=length(pvalue.Z))	stop("MAF is needed when hybridizing Z-score based meta-analysis with other methods")
			re<-Hybrid_Normal(pvalue=pvalue.Z,MAF=MAF.Z,CCsize=CCsize.Z)
			q_Normal<-re$q
			Var_Normal<-re$Var
		}
	} else {
		q_Normal<-0
		Var_Normal<-0
	}
	if(info[2]>0)
	{
		GCmat<-matrix(GCmat,length(pvalue.GC),2)
		re<-Hybrid_GC(pvalue=pvalue.GC,GCmat=GCmat,CCsize=CCsize.GC,Cutoff.GC=Cutoff.GC)	
		q_GC<-re$qjoint
		mu_GC<-re$mujoint
		g_GC<-re$gjoint
		NAset<-re$NAset
	} else {
		q_GC<-0
		mu_GC<-NULL
		g_GC<-NULL
		NAset<-NULL
	}
	if(!is.null(spldata))
	{
		if(is.vector(spldata))		spldata<-as.matrix(t(spldata))
		if(is.matrix(spldata))
		{
			k<-ncol(spldata)
			if(k%%3!=2)	stop("spldata should have (3k+2) number of columns where the first two columns represent Score and Var, next 3k represent k nodes, k values of K1, and k values of K2")
			if(sum(is.na(spldata))>0)	stop("Cannot have NAs in spldata, if there are different number of nodes for different studies, input spldata in list format")
			spldata<-split(spldata, seq(nrow(spldata)))
		}
		if(!is.list(spldata))
		{
			stop("spldata needs to be in one of the following formats: vector (only 1 study), matrix (all studies have same number of nodes), or list (different number of nodes in different studies)")
		}
	}

	return(Hybrid_meta(q_Normal,Var_Normal,q_GC,mu_GC,g_GC,NAset,spldata,Cutoff.meta)$p.value)
}


MetaSPA_Normal <-function(pvalue,MAF=NULL,CCsize)
{
	sgn<-sign(pvalue)
	pvalue<-abs(pvalue)
	if(is.vector(CCsize))	CCsize<-as.matrix(t(CCsize))
	ncases<-CCsize[,1]
	ncontrols<-CCsize[,2]
	Z<-sgn*qnorm(pvalue/2,lower.tail=FALSE)
	if(is.null(MAF))
	{
		Neff<-ncases*ncontrols/(ncases+ncontrols)
		Zmeta<-sum(sqrt(Neff)*Z)/sqrt(sum(Neff))
		pval<-pchisq(Zmeta^2,lower.tail=FALSE,df=1)
		return(sign(Zmeta)*pval)
	} else {
		Neff<-2*ncases*ncontrols*MAF*(1-MAF)/(ncases+ncontrols)
		Zmeta<-sum(sqrt(Neff)*Z)/sqrt(sum(Neff))
		pval<-pchisq(Zmeta^2,lower.tail=FALSE,df=1)
		return(sign(Zmeta)*pval)
	}

}


Hybrid_Normal<-function(pvalue,MAF,CCsize)
{
	sgn<-sign(pvalue)
	pvalue<-abs(pvalue)
	if(is.vector(CCsize))	CCsize<-as.matrix(t(CCsize))
	ncases<-CCsize[,1]
	ncontrols<-CCsize[,2]
	if(!is.null(MAF) & (length(MAF)!=length(pvalue) | length(which(is.na(MAF)))>0))	MAF=NULL
	Z<-sgn*qnorm(pvalue/2,lower.tail=FALSE)
	Neff<-2*ncases*ncontrols*MAF*(1-MAF)/(ncases+ncontrols)
	Zmeta<-sum(sqrt(Neff)*Z)
	Var<-sum(Neff)
	return(list(q=Zmeta,Var=Var))
}

Hybrid_GC<-function(pvalue,GCmat,CCsize,Cutoff.GC)
{
	truep<-abs(pvalue)
	sgn<-sign(pvalue)
	if(is.vector(CCsize))	CCsize<-as.matrix(t(CCsize))
	ncases<-CCsize[,1]
	ncontrols<-CCsize[,2]
	N2<-GCmat[,1]
	N1<-GCmat[,2]
	k<-length(ncases)
	qadj<-rep(NA,k)
	yjoint<-NULL
	mujoint<-NULL
	gjoint<-NULL
	if(length(Cutoff.GC)==1) Cutoff.GC<-rep(Cutoff.GC,k)
	NAset<-NULL
	for(s in 1:k)
	{
		n<-ncases[s]+ncontrols[s]
		mu<-rep(ncases[s]/n,n)
		y<-c(rep(1,ncases[s]),rep(0,ncontrols[s]))
		g<-c(rep(2,N2[s]),rep(1,N1[s]),rep(0,n-N2[s]-N1[s]))
		g1<-g-mean(g)
		m1<-mean(g)*ncases[s]
		var1<-mu[1]*(1-mu[1])*sum(g1^2)
		if(truep[s]>pnorm(Cutoff.GC[k],lower.tail=F)*2)
		{
			qadj[s]<-qnorm(truep[s]/2,lower.tail=F)*sgn[s]*sqrt(var1)
			yjoint<-c(yjoint,y)
			mujoint<-c(mujoint,mu)
			NAset<-c(NAset,length(gjoint)+N2[s]+N1[s]+1:(n-N2[s]-N1[s]))	
			gjoint<-c(gjoint,g1)
		} else {
			gcent<-0:2-mean(g)
			GenC=c(n-N2[s]-N1[s],N1[s],N2[s])
			rootadj<-try(uniroot(pval_MAC1,sgn[s]*c(0,2*sum(y)),p=mu[1],m1=0,var1=var1,GenC=GenC,gcent=gcent,truep[s],Cutoff=Cutoff.GC[s])$root,silent=T)
			if(class(rootadj) != "try-error")
			{
				qadj[s]<-rootadj	
				yjoint<-c(yjoint,y)
				mujoint<-c(mujoint,mu)
				NAset<-c(NAset,length(gjoint)+N2[s]+N1[s]+1:(n-N2[s]-N1[s]))	
				gjoint<-c(gjoint,g1)
			}
		}
	}
	qjoint<-sum(qadj,na.rm=T)
	return(list(qjoint=qjoint,mujoint=mujoint,gjoint=gjoint,NAset=NAset))
}

Hybrid_meta<-function(q_Normal,Var_Normal,q_GC,mu_GC,g_GC,NAset,spldata,Cutoff)
{

	gNA<-g_GC[NAset]
	muNA<-mu_GC[NAset]
	if(length(NAset)>0)
	{
		gNB<-g_GC[-NAset]
		muNB<-mu_GC[-NAset]
	} else {
		gNB=NULL
		muNB=NULL
	}
	var1<-sum(mu_GC * (1-mu_GC) * g_GC^2)
	NAmu= -sum(gNB*muNB)
	NAsigma=var1-sum(muNB*(1-muNB)*gNB^2)+Var_Normal

	if(is.null(spldata))
	{
		q_spl=0
		Var.spl=0
		nodes=NULL
		splfuny1=NULL
		splfuny2=NULL
	} else {
		k<-length(spldata)
		Score.spl<-rep(NA,k)
		Var.spl<-rep(NA,k)
		nodes<-list()
		splfuny1<-list()
		splfuny2<-list()
		for(s in 1:k)
		{
			nk<-length(spldata[[s]])
			if(nk%%3!=2)	stop("rows/list elements of spldata must have (3k+2) elements, where k is the number of nodes")
			nn<-(nk-2)/3
			Score.spl[s]<-spldata[[s]][1]
			Var.spl[s]<-spldata[[s]][2]
			nodes[[s]]<-spldata[[s]][2+1:nn]
			splfuny1[[s]]<-spldata[[s]][2+nn+1:nn]	
			splfuny2[[s]]<-spldata[[s]][2+2*nn+1:nn]
			int<-findInterval(0,nodes[[s]])
			nodes[[s]]<-c(nodes[[s]][1:int],0,nodes[[s]][(int+1):length(nodes[[s]])])
			splfuny1[[s]]<-c(splfuny1[[s]][1:int],0,splfuny1[[s]][(int+1):length(splfuny1[[s]])])
			splfuny2[[s]]<-c(splfuny2[[s]][1:int],Var.spl[s],splfuny2[[s]][(int+1):length(splfuny2[[s]])])
			dup<-which(duplicated(nodes[[s]]))
			nodes[[s]][dup]<-NA
			splfuny1[[s]][dup]<-NA
			splfuny2[[s]][dup]<-NA
		}
		q_spl=sum(Score.spl)
	}
	q<-q_Normal+q_GC+q_spl
	qinv = -q
	var2<-var1+sum(Var.spl)

	# Noadj
	pval.noadj<-pchisq(q^2/var2, lower.tail = FALSE, df=1)
	Is.converge=TRUE

	if(Cutoff < 10^-1){
		Cutoff=10^-1
	} 

	if(abs(q)/sqrt(var2) < Cutoff ){

		pval=pval.noadj
		
	} else {
		out.uni1<-getroot_K1_meta(0, q=q,gNB=gNB,muNB=muNB,NAmu=NAmu,NAsigma=NAsigma,nodes=nodes,splfuny1=splfuny1,splfuny2=splfuny2)
		out.uni2<-getroot_K1_meta(0, q=qinv,gNB=gNB,muNB=muNB,NAmu=NAmu,NAsigma=NAsigma,nodes=nodes,splfuny1=splfuny1,splfuny2=splfuny2)
		if(out.uni1$Is.converge==TRUE && out.uni2$Is.converge==TRUE)
		{
			p1<-Get_Saddle_Prob_meta(out.uni1$root, q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
			p2<-Get_Saddle_Prob_meta(out.uni2$root, qinv,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
			pval = abs(p1)+abs(p2)
			Is.converge=TRUE
		} else {
 			print("Error_Converge")
			pval<-pval.noadj
			Is.converge=FALSE	
		}		
		
	}
	
	if(pval!=0 && pval.noadj/pval>10^3)
	{
		return(Hybrid_meta(q_Normal,Var_Normal,q_GC,mu_GC,g_GC,NAset,spldata,Cutoff=Cutoff*2))
	} else {
		return(list(p.value=sign(q)*pval, p.value.NA=sign(q)*pval.noadj, Is.converge=Is.converge))
	}

}


Korg_meta<-function(t, gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		if(is.null(gNB))
		{
			out[i]<-NAmu*t1+0.5*NAsigma*t1^2
		} else {
			temp<-log(1 - muNB + muNB * exp(gNB* t1))
			out[i]<-sum(temp)+NAmu*t1+0.5*NAsigma*t1^2
		}
		if(!is.null(nodes))
		{
			out[i]<-out[i]+integrate(K1_adj_meta,0,t1,q=0,gNB=NULL,muNB=NULL,NAmu=0,NAsigma=0,nodes=nodes,splfuny1=splfuny1,splfuny2=splfuny2)$value
		}
	}
	return(out)
}


K1_adj_meta<-function(t, q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
{
	n.t<-length(t)	
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		temp3<-NAmu+NAsigma*t1
		if(is.null(gNB))
		{
			out[i]<-temp3-q
		} else {
			temp1<-(1 - muNB)* exp(-gNB * t1) + muNB
			temp2<-muNB *gNB
			out[i]<-sum(temp2/temp1)+temp3-q
		}
		if(!is.null(nodes))
		{
			k<-length(nodes)
			sval=0
			for(s in 1:k)
			{
				sfun<-splinefunH(na.omit(nodes[[s]]) , na.omit(splfuny1[[s]]) ,m=na.omit(splfuny2[[s]]))
				sval<-sval+sfun(t1,deriv=0)
			}
			out[i]<-out[i]+sval
		}
	}
	return(out)
}

K2_meta<-function(t, gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
{
	n.t<-length(t)
	out<-rep(0,n.t)
	
	for(i in 1:n.t){
		t1<-t[i]
		if(is.null(gNB))
		{
			out[i]<-NAsigma
		} else {
			temp1<-((1 - muNB)* exp(-gNB * t1) + muNB)^2
			temp2<-(1-muNB) * muNB * gNB^2 * exp(-gNB*t1)
			out[i]<-sum(temp2/temp1,na.rm=TRUE)+NAsigma
		}
		if(!is.null(nodes))
		{
			k<-length(nodes)
			var=0
			for(s in 1:k)
			{
				sfun<-splinefunH(na.omit(nodes[[s]]) , na.omit(splfuny1[[s]]) ,m=na.omit(splfuny2[[s]]))
				var<-var+sfun(t1,deriv=1)
			}
			out[i]<-out[i]+var
		}
	}
	return(out)
}

getroot_K1_meta<-function(init,q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2,tol=.Machine$double.eps^0.25,maxiter=1000)
{
		t<-init
		K1_eval<-K1_adj_meta(t,q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
		prevJump<- Inf
		rep<-1
		repeat
		{
			K2_eval<-K2_meta(t,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
			tnew<-t-K1_eval/K2_eval
			if(is.na(tnew))
			{
				conv=FALSE
				break
			}
			if(abs(tnew-t)<tol)
			{
				conv<-TRUE
				break
			}
			if(rep==maxiter)
			{
				conv<-FALSE
				break
			}

			newK1<-K1_adj_meta(tnew,q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
			if(sign(K1_eval)!=sign(newK1))
			{
				if(abs(tnew-t)>prevJump-tol)
				{
					tnew<-t+sign(newK1-K1_eval)*prevJump/2
					newK1<-K1_adj_meta(tnew,q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
					prevJump<-prevJump/2
				} else {
					prevJump<-abs(tnew-t)
				}
			}

			rep<-rep+1
			t<-tnew
			K1_eval<-newK1
		}
		return(list(root=t,n.iter=rep,Is.converge=conv))
}

Get_Saddle_Prob_meta<-function(zeta, q,gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
{
	k1<-Korg_meta(zeta,  gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
	k2<-K2_meta(zeta,  gNB,muNB,NAmu,NAsigma,nodes,splfuny1,splfuny2)
	
	if(is.finite(k1) && is.finite(k2))
	{
	temp1<-zeta * q - k1

	
	w<-sign(zeta) * (2 *temp1)^{1/2}
	v<- zeta * (k2)^{1/2}
			
	Z.test<-w + 1/w * log(v/w)	

	if(Z.test > 0){
		pval<-pnorm(Z.test, lower.tail = FALSE)
	} else {
		pval= -pnorm(Z.test, lower.tail = TRUE)
	}	
	} else {
	pval<-0
	}
	
	return(pval)
}





