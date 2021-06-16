int main() {
	scanf("%d", &N);
	for(; N --;) {
		scanf("%d", &A);
		printf("%d\n", acSearch(A, 2));
	}
	return 0;
}
