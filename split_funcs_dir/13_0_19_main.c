void main()
{
	int f(int x,int m);
	int k,i,j,n,sum=0;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		scanf("%d",&k);
		for(j=2;j<=k;j++)
		{
			if(k%j==0)
			{
				sum+=f(k,j);
			}
		}
		printf("%d\n",sum);
		sum=0;
	}
}
