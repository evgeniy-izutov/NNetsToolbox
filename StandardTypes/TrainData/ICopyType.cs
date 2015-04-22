namespace StandardTypes {
	public interface ICopyType<out T> {
		T Copy();
	}
}
