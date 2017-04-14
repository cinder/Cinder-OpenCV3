#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "CinderOpenCV.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class ocvKalmanApp : public App {
  public:
	void setup() override;
	void mouseDrag( MouseEvent event ) override;
	void update() override;
	void draw() override;
    
    class KalmanFilter {
        
      protected:
        cv::KalmanFilter    mKF;
        cv::Mat_<float>     mMeasurement;
        cv::Point           mPrediction;
        
      public:
    
        KalmanFilter( cv::Point initialPt )
        {
            mKF = cv::KalmanFilter( 4, 2, 0 );
            mKF.transitionMatrix = ( cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1 );
            mMeasurement = cv::Mat_<float>( 2, 1 );
            mMeasurement.setTo( cv::Scalar( 0 ) );
        
            // init...
            mKF.statePre.at<float>( 0 ) = initialPt.x;
            mKF.statePre.at<float>( 1 ) = initialPt.y;
            mKF.statePre.at<float>( 2 ) = 0;
            mKF.statePre.at<float>( 3 ) = 0;
            setIdentity( mKF.measurementMatrix );
            setIdentity( mKF.processNoiseCov, cv::Scalar::all( 1e-4 ) );
            setIdentity( mKF.measurementNoiseCov, cv::Scalar::all( 1e-1 ) );
            setIdentity( mKF.errorCovPost, cv::Scalar::all( .1 ));
            
            mPrediction = initialPt;

            // prime the filter, otherwise it thinks it's at 0,0
            correct( initialPt );
        }
        
        cv::Point updatePrediction()
        {
            cv::Mat prediction = mKF.predict();
            mPrediction = cv::Point( prediction.at<float>(0), prediction.at<float>( 1 ) );
            
            return mPrediction;
        }
        
        cv::Point correct( cv::Point pt )
        {
            // update point
            mMeasurement( 0 ) = pt.x;
            mMeasurement( 1 ) = pt.y;
            
            cv::Point measPt( mMeasurement( 0 ), mMeasurement( 1 ) );
            
            // The "correct" phase that is going to use the predicted value and our measurement
            cv::Mat estimated = mKF.correct( mMeasurement );
            cv::Point statePt( estimated.at<float>( 0 ), estimated.at<float>( 1 ) );
            
            mPrediction = statePt;
            
            return mPrediction;
        }
    }; // class KalmanFilter
    
    KalmanFilter *mFilter;
    vector<vec2> mMousePoints;
    vector<vec2> mKalmanPoints;
    
};  // class ocvKalmanApp


void ocvKalmanApp::setup()
{
    setFrameRate( 30.0f );
}

void ocvKalmanApp::mouseDrag( MouseEvent event )
{
    mMousePoints.push_back( event.getPos() );
    
    if( mKalmanPoints.empty() ) {
        mFilter = new KalmanFilter( toOcv( event.getPos() ) );
        mKalmanPoints.push_back( event.getPos() );
    } else {
        mFilter->correct( toOcv( event.getPos() ) );
        mKalmanPoints.push_back( fromOcv( mFilter->updatePrediction() ) );
        console() << mKalmanPoints.back() << endl;
    }
}

void ocvKalmanApp::update()
{
}

void ocvKalmanApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
    
    gl::color( 1.0f, 0.0f, 0.0f );
    gl::begin( GL_LINE_STRIP );
    for( const vec2 &point : mMousePoints ) {
        gl::vertex( point );
    }
    gl::end();
    
    gl::color( 1.0f, 1.0f, 0.0f );
    gl::begin( GL_LINE_STRIP );
    for( const vec2 &point : mKalmanPoints ) {
        gl::vertex( point );
    }
    gl::end();
}

CINDER_APP( ocvKalmanApp, RendererGl )
