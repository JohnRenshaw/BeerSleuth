// https://angular-ui.github.io/

// setup app and pass ui.bootstrap as dep
var myApp = angular.module("angularTypeahead", ["ui.bootstrap"]);

// define factory for data source
myApp.factory("Beers", function(){
  var states = {{beer_list}} ;
  return states;
  
});

// setup controller and pass data source
myApp.controller("TypeaheadCtrl", function($scope, Beers) {
	
	$scope.selected = undefined;
	
	$scope.beers = Beers;
	
});