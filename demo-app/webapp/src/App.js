import React, { Component } from 'react';
import axios from 'axios';
import StarRatings from 'react-star-ratings';

const ENDPOINT = "https://somelink.com/"

class App extends Component {
  constructor(props){
    super(props);
    this.state = {
      review: "",
      rating: 0,
      err: false
    }
  }
  submitReview = (e) => {
    e.preventDefault()
    const data = {
      review: this.state.review
    }
    if (data.review !== '' && data.review !== null){
      axios.post(ENDPOINT, data)
      .then(resp => this.setState({rating:resp.data.data.class}))
      .catch(err => this.setState({err:err}))
    }
  }

  render() {
    return (
      <div className="App container">
        <div className="jumbotron jumbotron-fluid mt-5">
          <div className="container">
            <h1 className="display-4">Rate Prediction App</h1>
            <p className="lead">This is my Machine Learning project, that predicts the rate of reviews.</p>
          </div>
        </div>
        <form>
          <div className="form-group">
            <textarea className="form-control" placeholder="Write a review..." required onChange={(e) => this.setState({review:e.target.value})}></textarea>
          </div>
          <button type="submit" className="btn btn-primary float-right" onClick={this.submitReview}>Submit</button>
        </form>
        <div className="d-flex justify-content-center m-5">
          {this.state.err ? (
            <div className="text-center">
              <div class="alert alert-danger" role="alert">
                Something went wrong
              </div>
            </div>
          ) : (
            <StarRatings
              rating={this.state.rating}
              starRatedColor="blue"
              numberOfStars={5}
              name='rating'
            />
          )}
          
        </div>
      </div>
    );
  }
}

export default App;