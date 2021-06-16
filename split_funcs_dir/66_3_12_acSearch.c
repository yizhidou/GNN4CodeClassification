int acSearch(int n, int i) {
	int ret = 0;

	if(n == 1) ret = 1;
	else for(; i <= n; ++ i)
		if(n%i == 0) ret += acSearch(n/i, i);

	return ret;
}
