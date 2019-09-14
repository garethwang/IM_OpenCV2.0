#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

#define INDEX(i,j) ((i) * m_cols + (j))

namespace kvld {
	// Forward declaration, definition below
	template <typename T> class matrix;

	/// Matrix class
	template <typename T>
	class matrix
	{
	public:
		static matrix<T> ones(int m) { return ones(m, m); }
		static matrix<T> ones(int m, int n) {
			matrix<T> M(m, n);
			for (int i = M.nElements() - 1; i >= 0; i--)
				M.p[i] = (T)1;
			return M;
		};

	public:
		matrix(int m, int n) {alloc(m, n);};
		matrix(const matrix<T>& m) {
			alloc(m.m_rows, m.m_cols);
			for (int i = nElements() - 1; i >= 0; i--)
				p[i] = m.p[i];
		};
		virtual ~matrix(){free();};

		T  operator() (int i, int j) const {
			assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
			return p[INDEX(i, j)];
		};
		T& operator() (int i, int j) {
			assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
			return p[INDEX(i, j)];
		};

		matrix<T> operator*(T a) const {
			matrix<T> prod(m_rows, m_cols);
			for (int i = nElements() - 1; i >= 0; i--)
				prod.p[i] = a * p[i];
			return prod;
		};


	protected:
		int m_rows; ///< Number of rows.
		int m_cols; ///< Number of columns.
		T* p; ///< 1-D array of coefficients.

		void alloc(int m, int n) {
			assert(m > 0 && n > 0);
			m_rows = m;
			m_cols = n;
			p = new T[m*n];
		}; ///< Allocate the array value.
		void free(){
			delete[] p;
			p = NULL;
		}; ///< Free the array value.
		int nElements() const {
			return m_rows*m_cols;}; ///< Number of elements in the matrix.
	}; // class matrix
}

typedef std::pair<size_t, size_t> Pair;
const float PI = 4.0f * atan(1.0f);

//-- Container for a 2D image
//-- This class ensure that the image have a width and a height
//-- and a 2D array of T data.
//-
//-- Data is saved in row major format
//-- Pixel access is done with operator(y,x)
//  [2/3/2011 pierre MOULON]
//---------------------------

template <typename T>
class Image
{

public:
	typedef T Tpixel;	//-- Store the pixel data type

	//------------------------------
	//-- Image construction method
	Image(){ _data = NULL; _width = _height = 0; }

	Image(size_t width, size_t height, const T val = T())
	{
		_height = height;
		_width = width;
		_data = new T[_width * _height];
		memset(_data, static_cast<int>(val), sizeof(T) * _width * _height);
	}

	Image(size_t width, size_t height, const T * imdata){
		_height = height;
		_width = width;
		_data = new T[_width * _height];
		memcpy(_data, imdata, _width * _height * sizeof(T));
	}

	Image(const Image<T> & I){
		_data = NULL; _width = _height = 0;
		(*this) = I;
	}

	Image& operator=(const Image<T> & I) {

		if (this != &I)
		{
			_height = I._height;
			_width = I._width;
			if (_data)
				delete[] _data;
			_data = new T[_width * _height];
			memcpy(_data, I._data, _width * _height * sizeof(T));
		}
		return (*this);
	}

	virtual ~Image(){
		if (_data)
			delete[] _data;
		_width = _height = 0;
	}
	//-- Image construction method
	//------------------------------


	//------------------------------
	//-- accessors/getters methods
	/// Retrieve the width of the image
	size_t Width()  const { return _width; }
	/// Retrieve the height of the image
	size_t Height() const { return _height; }
	/// Return the depth in byte of the pixel (unsigned char will return 1)
	size_t Depth() const  { return sizeof(Tpixel); }

	/// image memory access
	T * data() const { return _data; }
	/// random pixel access
	inline const T & operator()(size_t y, size_t x) const { return _data[_width * y + x]; }
	/// random pixel access
	inline T & operator()(size_t y, size_t x) { return _data[_width * y + x]; }

	inline bool operator==(const Image<T> & img) const
	{
		if (_width == img._width &&
			_height == img._height)
			return (0 == memcmp(_data, img._data, Width()*Height()*sizeof(T)));
		else
			return false;
	}
	//-- accessors/getters methods
	//------------------------------

	/// Tell if a point is inside the image.
	bool Contains(size_t y, size_t x) const {
		return x < _width && y < _height;
	}

	//-- Utils
	//
	void Resize(size_t width, size_t height){
		_height = height;
		_width = width;
		if (_data)
			delete[] _data;
		_data = new T[_width * _height];
	}
	void fill(const T & val)
	{
		memset(_data, static_cast<int>(val), sizeof(T) * _width * _height);
	}

	friend std::ostream & operator<<(std::ostream & os, const Image<T> & im) {
		os << im.Width() << " " << im.Height();
		for (size_t j = 0; j < im.Height(); ++j)
		{
			for (size_t i = 0; i < im.Width(); ++i)
				os << im(j, i) << " ";
			os << std::endl;
		}
		return os;
	}

protected:
	//-- Internal data :
	// None
	T * _data;
	size_t _width;
	size_t _height;
};

// BASIC STRUCTURES:

// Keypoints:
#define OriSize  8
#define IndexSize  4
#define VecLength  IndexSize * IndexSize * OriSize

/* Keypoint structure:
	position:	x,y
	scale:		s
	orientation:	angle
	descriptor:	array of gradient orientation histograms in a neighbors */
struct keypoint {
	float x, y, scale, angle;
	float vec[VecLength];
};

/// Save matching position between two points.
struct Match
{
    Match() {}
    Match(float ix1, float iy1, float ix2, float iy2)
    : x1(ix1), y1(iy1), x2(ix2), y2(iy2) {}
    float x1, y1, x2, y2;

    /**
    * Load the corresponding matches from file.
    * \param nameFile   The file where matches were saved.
    * \param vec_match  The loaded corresponding points.
    * \return bool      True if everything was ok, otherwise false.
    */
    static bool loadMatch(const char* nameFile, std::vector<Match>& vec_match)
    {
      vec_match.clear();
      std::ifstream f(nameFile);
      while( f.good() ) {
        std::string str;
        std::getline(f, str);
        if( f.good() ) {
          std::istringstream s(str);
          Match m;
          s >> m;
          if(!s.fail() )
            vec_match.push_back(m);
        }
      }
      return !vec_match.empty();
    }
    
    /**
    * Save the corresponding matches to file.
    * \param nameFile   The file where matches will be saved.
    * \param vec_match  The matches that we want to save.
    * \return bool True if everything was ok, otherwise false.
    */
    static bool saveMatch(const char* nameFile, const std::vector<Match>& vec_match)
    {
      std::ofstream f(nameFile);
      if( f.is_open() ) {
        std::vector<Match>::const_iterator it = vec_match.begin();
        for(; it != vec_match.end(); ++it)
          f << *it;
      }
      return f.is_open();
    }

    /// Lexicographical ordering of matches. Used to remove duplicates.
    friend bool operator<(const Match& m1, const Match& m2)
    {
      if(m1.x1 < m2.x1) return true;
      if(m1.x1 > m2.x1) return false;

      if(m1.y1 < m2.y1) return true;
      if(m1.y1 > m2.y1) return false;

      if(m1.x2 < m2.x2) return true;
      if(m1.x2 > m2.x2) return false;

      return (m1.y2 < m2.y2);
    }

    /// Comparison Operator
    friend bool operator==(const Match& m1, const Match& m2)
    {
      return (m1.x1==m2.x1 && m1.y1==m2.y1 &&
              m1.x2==m2.x2 && m1.y2==m2.y2);
    }

    friend std::ostream& operator<<(std::ostream& os, const Match & m)
    {
      return os << m.x1 << " " << m.y1 << " "
        << m.x2 << " " << m.y2 << std::endl;
    }

    friend std::istream& operator>>(std::istream & in, Match & m)
    {
      return in >> m.x1 >> m.y1 >> m.x2 >> m.y2;
    }
};

#endif // DATA_TYPE_H