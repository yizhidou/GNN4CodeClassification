int f(int x,int m)
{
	int i,sum=0;
	if(m==x)
		sum=1;
	else
	{
		x=x/m;
		for(i=m;i<=x;i++)
		{
			if(x%i==0)
			{
				sum+=f(x,i);
			}
		}
	}
	return sum;
}